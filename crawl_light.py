# crawl_light.py
# Light++ crawler voor Aseon (BFS, robots-respect, caps, issues & quick_wins, social detection)
# Doel: stabiel op 2GB/1CPU, snelle signalen voor SEO/AEO/GEO agents.

import os
import re
import time
import html
import json
import queue
import urllib.parse as urlparse
from dataclasses import dataclass
from typing import List, Dict, Any, Set, Tuple, Optional

import requests
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser


# ---------- Config via env ----------
REQ_TIMEOUT = float(os.getenv("CRAWL_REQ_TIMEOUT_SEC", "6"))
CONTENT_CAP_BYTES = int(os.getenv("CRAWL_CONTENT_CAP_BYTES", "150000"))  # 150 KB
MAX_QUEUE = int(os.getenv("CRAWL_MAX_QUEUE", "200"))
HARD_MAX_PAGES = int(os.getenv("CRAWL_MAX_PAGES_HARD", "20"))

DEFAULT_UA = os.getenv(
    "CRAWL_UA",
    "AseonBot/0.3 (+https://aseon.ai; support@aseon.ai)"
)

# Host matching
ALLOW_WWW_EQUIV = os.getenv("CRAWL_ALLOW_WWW_EQUIV", "true").lower() == "true"
ALLOW_SUBDOMAINS = os.getenv("CRAWL_ALLOW_SUBDOMAINS", "false").lower() == "true"

# Social whitelist (comma-separated domains allowed, we match by suffix)
_DEFAULT_SOCIALS = [
    "twitter.com",
    "x.com",
    "linkedin.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "youtube.com",
    "github.com",
    "gitlab.com",
    "bitbucket.org",
    "producthunt.com",
    "crunchbase.com",
    "wikidata.org",
    "wikipedia.org"
]
SOCIAL_WHITELIST = [
    d.strip().lower() for d in os.getenv("CRAWL_SOCIAL_WHITELIST", ",".join(_DEFAULT_SOCIALS)).split(",")
    if d.strip()
]


@dataclass
class PageResult:
    url: str
    final_url: str
    status: int
    redirect_chain: List[str]
    title: Optional[str]
    meta_description: Optional[str]
    canonical: Optional[str]
    h1: Optional[str]
    h2: List[str]
    h3: List[str]
    noindex: Optional[bool]
    nofollow: Optional[bool]
    internal_links: int
    issues: List[str]


def _netloc_key(netloc: str) -> str:
    """Normaliseer host: lowercase, strip poort, strip leidende 'www.' voor vergelijking."""
    nl = (netloc or "").strip().lower()
    if not nl:
        return nl
    if ":" in nl:
        nl = nl.split(":", 1)[0]
    if ALLOW_WWW_EQUIV and nl.startswith("www."):
        nl = nl[4:]
    return nl


def _same_site(u: str, root_netloc_key: str) -> bool:
    """Zelfde site? Exacte hostmatch na normalisatie, of (optioneel) subdomain-allow."""
    try:
        nl = _netloc_key(urlparse.urlparse(u).netloc)
        if not nl:
            return False
        if nl == root_netloc_key:
            return True
        if ALLOW_SUBDOMAINS and nl.endswith("." + root_netloc_key):
            return True
        return False
    except Exception:
        return False


def _norm_url(u: str) -> str:
    try:
        parsed = urlparse.urlparse(u)
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc
        path = parsed.path or "/"
        return urlparse.urlunparse((scheme, netloc, path, "", parsed.query, ""))
    except Exception:
        return u


def _extract_meta_robots(soup: BeautifulSoup) -> Tuple[Optional[bool], Optional[bool]]:
    """
    Return (noindex, nofollow) from <meta name="robots"> or <meta name="googlebot"> where present.
    None = niet gevonden.
    """
    def parse_meta(val: str) -> Tuple[Optional[bool], Optional[bool]]:
        v = (val or "").lower()
        if not v:
            return (None, None)
        toks = [t.strip() for t in v.split(",")]
        ni = True if "noindex" in toks else (False if "index" in toks else None)
        nf = True if "nofollow" in toks else (False if "follow" in toks else None)
        return (ni, nf)

    tag = soup.find("meta", attrs={"name": re.compile(r"^(robots|googlebot)$", re.I)})
    if tag and tag.get("content"):
        return parse_meta(tag.get("content"))

    return (None, None)


def _safe_text(x: Optional[str], cap: int) -> Optional[str]:
    if not x:
        return None
    x = html.unescape(x).strip()
    if not x:
        return None
    return x[:cap]


def _iter_links(soup: BeautifulSoup, base: str) -> List[str]:
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#"):
            continue
        absu = urlparse.urljoin(base, href)
        out.append(absu)
    return out


def _get_session(ua: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": ua, "Accept": "text/html,application/xhtml+xml"})
    return s


def _fetch_capped(session: requests.Session, url: str) -> Tuple[int, bytes, List[str], str]:
    """
    Haal URL op met redirect-volgorde en content cap (bytes).
    Retourneert (status, body_bytes_capped, redirect_chain, final_url).
    """
    chain: List[str] = []
    try:
        current = url
        for _ in range(6):  # max redirects
            resp = session.get(current, timeout=REQ_TIMEOUT, allow_redirects=False, stream=True)
            status = resp.status_code
            chain.append(_norm_url(current))
            if status in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
                loc = urlparse.urljoin(current, resp.headers["Location"])
                current = loc
                try:
                    resp.close()
                except Exception:
                    pass
                continue

            body = bytearray()
            for chunk in resp.iter_content(chunk_size=16384):
                if chunk:
                    if len(body) + len(chunk) > CONTENT_CAP_BYTES:
                        body.extend(chunk[: max(0, CONTENT_CAP_BYTES - len(body))])
                        break
                    body.extend(chunk)
            final_url = _norm_url(resp.url or current)
            try:
                resp.close()
            except Exception:
                pass
            return status, bytes(body), chain, final_url

        return 310, b"", chain, _norm_url(current)
    except requests.RequestException:
        return 0, b"", chain, _norm_url(url)
    except Exception:
        return 0, b"", chain, _norm_url(url)


def _parse_html(url: str, body: bytes) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], List[str], List[str]]:
    """
    Extract: title, meta_description, canonical, h1, h2[], h3[].
    """
    if not body:
        return (None, None, None, None, [], [])

    soup = BeautifulSoup(body, "html.parser")

    title = _safe_text(soup.title.string if soup.title else None, 300)

    meta = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    meta_desc = _safe_text(meta.get("content") if meta else None, 420)

    link_canon = soup.find("link", attrs={"rel": re.compile(r"(^|\s)canonical(\s|$)", re.I)})
    canonical = _safe_text(urlparse.urljoin(url, link_canon.get("href")) if link_canon and link_canon.get("href") else None, 1000)

    h1_tag = soup.find("h1")
    h1 = _safe_text(h1_tag.get_text(" ", strip=True) if h1_tag else None, 300)

    def tops(sel: str, cap_each: int, limit: int) -> List[str]:
        arr = []
        for el in soup.select(sel)[:limit]:
            txt = _safe_text(el.get_text(" ", strip=True), cap_each)
            if txt:
                arr.append(txt)
        return arr

    h2 = tops("h2", 240, 30)
    h3 = tops("h3", 200, 40)

    return (title, meta_desc, canonical, h1, h2, h3)


def _detect_issues(p: PageResult) -> List[str]:
    issues: List[str] = []
    if p.status in (301, 302, 303, 307, 308):
        issues.append("redirect")
    if p.status >= 400 and p.status < 600:
        issues.append("http_error")
    if not p.title:
        issues.append("missing_title")
    if not p.meta_description:
        issues.append("missing_meta_description")
    if not p.h1:
        issues.append("missing_h1")
    if p.noindex is True:
        issues.append("noindex")
    if p.nofollow is True:
        issues.append("nofollow")
    if p.canonical and p.canonical != p.final_url:
        issues.append("canonical_differs")
    return issues


def _summarize(pages: List[PageResult], started_ms: float, capped_by_runtime: bool) -> Dict[str, Any]:
    ok_200 = sum(1 for p in pages if p.status == 200)
    redirs = sum(1 for p in pages if 300 <= p.status < 400)
    errs = sum(1 for p in pages if p.status >= 400 or p.status == 0)
    duration_ms = int((time.time() - started_ms) * 1000)
    return {
        "pages_total": len(pages),
        "ok_200": ok_200,
        "redirect_3xx": redirs,
        "errors_4xx_5xx": errs,
        "fetch_errors": sum(1 for p in pages if p.status == 0),
        "duration_ms": duration_ms,
        "capped_by_runtime": capped_by_runtime
    }


def _build_quick_wins(pages: List[PageResult]) -> List[Dict[str, Any]]:
    wins: List[Dict[str, Any]] = []
    if any("redirect" in p.issues for p in pages):
        wins.append({"type": "redirects"})
    if any("missing_meta_description" in p.issues for p in pages):
        wins.append({"type": "missing_meta_description"})
    if any("missing_title" in p.issues for p in pages):
        wins.append({"type": "missing_title"})
    if any("missing_h1" in p.issues for p in pages):
        wins.append({"type": "missing_h1"})
    if any("noindex" in p.issues for p in pages):
        wins.append({"type": "noindex_present"})
    if any("http_error" in p.issues for p in pages):
        wins.append({"type": "http_errors"})
    return wins


def _is_social_domain(netloc: str) -> bool:
    nl = (netloc or "").lower()
    if ":" in nl:
        nl = nl.split(":", 1)[0]
    if nl.startswith("www."):
        nl = nl[4:]
    return any(nl == d or nl.endswith("." + d) for d in SOCIAL_WHITELIST)


def _canonical_social_url(u: str) -> Optional[str]:
    try:
        p = urlparse.urlparse(u)
        if not _is_social_domain(p.netloc):
            return None
        # strip query/fragment; normalize path basic
        path = p.path or "/"
        # drop trailing slashes (except root)
        if len(path) > 1 and path.endswith("/"):
            path = path[:-1]
        # heuristics: drop obvious share/intent paths
        bad_segments = ("/share", "/intent", "/oauth", "/login", "/signup")
        if any(seg in path.lower() for seg in bad_segments):
            return None
        return urlparse.urlunparse(("https", p.netloc.lower(), path, "", "", ""))
    except Exception:
        return None


def crawl_site(start_url: str, max_pages: int = 10, ua: str = DEFAULT_UA) -> Dict[str, Any]:
    """
    BFS crawl (same-site) met robots-respect, caps, basis-issues + social detectie.
    Return shape:
    {
      "start_url": ...,
      "pages": [ PageResult as dict ... ],
      "summary": {...},
      "quick_wins": [...],
      "social_profiles": ["https://twitter.com/...", ...]
    }
    """
    max_pages = max(1, min(int(max_pages), HARD_MAX_PAGES))
    started = time.time()

    # Normalize start_url
    if not start_url.startswith(("http://", "https://")):
        start_url = "https://" + start_url
    parsed_root = urlparse.urlparse(start_url)
    root = _norm_url(start_url)
    root_netloc_key = _netloc_key(parsed_root.netloc)

    # Robots
    robots_url = urlparse.urljoin(root, "/robots.txt")
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        pass  # bij failure: toestaan

    session = _get_session(ua)

    seen: Set[str] = set()
    q: "queue.Queue[str]" = queue.Queue()
    q.put(root)
    seen.add(root)

    pages: List[PageResult] = []
    capped_by_runtime = False
    social_profiles: Set[str] = set()

    def enqueue(u: str):
        if len(seen) >= MAX_QUEUE:
            return
        nu = _norm_url(u)
        if nu in seen:
            return
        if not _same_site(nu, root_netloc_key):
            return
        seen.add(nu)
        try:
            q.put_nowait(nu)
        except queue.Full:
            pass

    while len(pages) < max_pages and not q.empty():
        url = q.get_nowait()

        # robots check
        try:
            if not rp.can_fetch(ua, url):
                pages.append(PageResult(
                    url=url,
                    final_url=url,
                    status=999,  # pseudo: robots blocked
                    redirect_chain=[],
                    title=None,
                    meta_description=None,
                    canonical=None,
                    h1=None,
                    h2=[],
                    h3=[],
                    noindex=None,
                    nofollow=None,
                    internal_links=0,
                    issues=["robots_blocked"]
                ))
                continue
        except Exception:
            pass

        status, body, chain, final_url = _fetch_capped(session, url)

        title = meta_desc = canonical = h1 = None
        h2: List[str] = []
        h3: List[str] = []
        noindex = nofollow = None
        internal_links = 0

        if status == 200 and body:
            # parse
            title, meta_desc, canonical, h1, h2, h3 = _parse_html(final_url or url, body)
            soup = BeautifulSoup(body, "html.parser")
            # robots meta
            ni, nf = _extract_meta_robots(soup)
            noindex, nofollow = ni, nf
            # links
            links = _iter_links(soup, final_url or url)
            internal_links = 0
            for l in links:
                if _same_site(l, root_netloc_key):
                    internal_links += 1
                    enqueue(l)
                # social detectie (ook als het externe links zijn)
                soc = _canonical_social_url(l)
                if soc:
                    social_profiles.add(soc)

        # bouw page result
        page = PageResult(
            url=url,
            final_url=final_url or url,
            status=status,
            redirect_chain=chain,
            title=title,
            meta_description=meta_desc,
            canonical=canonical,
            h1=h1,
            h2=h2,
            h3=h3,
            noindex=noindex,
            nofollow=nofollow,
            internal_links=internal_links,
            issues=[]
        )
        page.issues = _detect_issues(page)
        pages.append(page)

    # Summary & quick wins
    summary = _summarize(pages, started, capped_by_runtime)
    quick_wins = _build_quick_wins(pages)

    # Convert PageResult to dicts
    pages_out: List[Dict[str, Any]] = []
    for p in pages:
        pages_out.append({
            "url": p.url,
            "final_url": p.final_url,
            "status": p.status,
            "redirect_chain": p.redirect_chain,
            "title": p.title,
            "meta_description": p.meta_description,
            "canonical": p.canonical,
            "h1": p.h1,
            "h2": p.h2,
            "h3": p.h3,
            "noindex": p.noindex,
            "nofollow": p.nofollow,
            "internal_links": p.internal_links,
            "issues": p.issues
        })

    return {
        "start_url": root,
        "pages": pages_out,
        "summary": summary,
        "quick_wins": quick_wins,
        "social_profiles": sorted(list(social_profiles))
    }
