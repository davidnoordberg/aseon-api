# crawl_light.py
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import Dict, Any, List, Set, Tuple
from collections import deque

DEFAULT_UA = "AseonBot/0.2 (+https://aseon.ai)"
MAX_REDIRECTS = 5
MAX_CONTENT_BYTES = 1_500_000  # ~1.5 MB per page
REQ_TIMEOUT = 10
MAX_QUEUE_SIZE = 2000  # safeguard tegen runaway crawls

def _registrable_host(netloc: str) -> str:
    n = (netloc or "").lower()
    return n[4:] if n.startswith("www.") else n

def _same_site(u1: str, u2: str) -> bool:
    a = urlparse(u1); b = urlparse(u2)
    return _registrable_host(a.netloc) == _registrable_host(b.netloc)

def _head_ok(url: str, ua: str) -> Tuple[int, int]:
    """HEAD om grootte te schatten; retourneert (status, content_length_of_guess)."""
    try:
        r = requests.head(url, headers={"User-Agent": ua}, timeout=REQ_TIMEOUT, allow_redirects=True)
        cl = r.headers.get("Content-Length")
        size = int(cl) if cl and cl.isdigit() else -1
        status = r.status_code
        r.close()
        return status, size
    except Exception:
        return 0, -1

def _get_with_cap(url: str, ua: str) -> Tuple[int, str, str, List[str]]:
    """
    GET met redirect-follow; content capped op MAX_CONTENT_BYTES.
    Return: (status, final_url, text_capped, redirect_chain)
    """
    redirect_chain: List[str] = []
    try:
        with requests.Session() as s:
            s.max_redirects = MAX_REDIRECTS
            resp = s.get(url, headers={"User-Agent": ua}, timeout=REQ_TIMEOUT, allow_redirects=True, stream=True)
            try:
                final_url = resp.url
                status = resp.status_code
                # bouw text tot cap
                text_parts: List[bytes] = []
                read = 0
                if status == 200:
                    for chunk in resp.iter_content(chunk_size=32_768):
                        if not chunk:
                            break
                        read += len(chunk)
                        if read > MAX_CONTENT_BYTES:
                            # stop lezen, we hebben genoeg
                            break
                        text_parts.append(chunk)
                # history → redirect chain
                for h in resp.history[:MAX_REDIRECTS]:
                    # prefer Location header; fallback naar h.url
                    redirect_chain.append(h.headers.get("Location") or h.url)
                text = b"".join(text_parts).decode(resp.encoding or "utf-8", errors="replace") if text_parts else ""
                return status, final_url, text, redirect_chain
            finally:
                resp.close()
    except requests.exceptions.TooManyRedirects:
        return 310, url, "", ["too_many_redirects"]
    except Exception:
        return 0, url, "", []

def _allowed_by_robots(base_url: str, robots_txt: str, path: str) -> bool:
    # minimalistische robots: Disallow regels voor '*' of 'AseonBot'
    consider = False
    for line in (robots_txt or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("user-agent:"):
            ua = line.split(":", 1)[1].strip().lower()
            consider = (ua in ("*", "aseonbot", "aseonbot/0.2 (+https://aseon.ai)"))
        elif consider and line.lower().startswith("disallow:"):
            rule = line.split(":", 1)[1].strip()
            rel = path or "/"
            if rule and rel.startswith(rule):
                return False
    return True

def _load_robots(start_url: str, ua: str) -> Tuple[str, str]:
    """Return (robots_base, robots_txt). Probeert www/non-www varianten."""
    pu = urlparse(start_url)
    host = _registrable_host(pu.netloc)
    candidates = [
        f"{pu.scheme}://{pu.netloc}/robots.txt",
        f"{pu.scheme}://www.{host}/robots.txt",
        f"{pu.scheme}://{host}/robots.txt",
    ]
    for u in candidates:
        try:
            r = requests.get(u, headers={"User-Agent": ua}, timeout=6)
            try:
                if r.status_code == 200 and len(r.text) < 200_000:
                    return f"{pu.scheme}://{pu.netloc}", r.text
            finally:
                r.close()
        except Exception:
            pass
    return f"{pu.scheme}://{pu.netloc}", ""

def _extract_meta_robots(soup: BeautifulSoup) -> Tuple[bool, bool]:
    noindex = nofollow = False
    tag = soup.find("meta", attrs={"name": re.compile("^robots$", re.I)})
    if tag and tag.get("content"):
        c = tag["content"].lower()
        if "noindex" in c: noindex = True
        if "nofollow" in c: nofollow = True
    return noindex, nofollow

def crawl_site(start_url: str, max_pages: int = 20, ua: str = DEFAULT_UA) -> Dict[str, Any]:
    """
    Memory-safe lightweight crawler:
    - Volgt redirects (begrensd), cap’t response body per pagina.
    - Respecteert robots (basic).
    - Parse: title, meta description, H1–H3, canonical, noindex/nofollow.
    - Blijft binnen hetzelfde registrable host (www/non-www gelijk).
    - Bounded queue & content, zodat RAM niet oploopt.
    """
    # normaliseer start_url
    start_url = start_url.strip()
    if not start_url.startswith(("http://", "https://")):
        start_url = "https://" + start_url
    if not start_url.endswith("/"):
        start_url += "/"

    q: deque[str] = deque([start_url])
    visited: Set[str] = set()
    pages: List[Dict[str, Any]] = []
    t0 = time.time()

    robots_base, robots_txt = _load_robots(start_url, ua)

    count = 0
    while q and count < max_pages:
        url = q.popleft()
        if url in visited:
            continue
        visited.add(url)
        count += 1

        issues: List[str] = []
        pu = urlparse(url)
        # robots check
        try:
            if not _allowed_by_robots(robots_base, robots_txt, pu.path):
                pages.append({
                    "url": url, "final_url": url, "status": 999,
                    "redirect_chain": [], "issues": ["blocked_by_robots"],
                    "title": None, "meta_description": None, "h1": None,
                    "h2": [], "h3": [], "canonical": None,
                    "noindex": None, "nofollow": None
                })
                continue
        except Exception:
            pass

        # HEAD check: skip super grote pagina’s
        st_head, size_guess = _head_ok(url, ua)
        if st_head >= 400:
            pages.append({
                "url": url, "final_url": url, "status": st_head,
                "redirect_chain": [], "issues": [f"http_{st_head}"],
                "title": None, "meta_description": None, "h1": None,
                "h2": [], "h3": [], "canonical": None,
                "noindex": None, "nofollow": None
            })
            continue
        if size_guess > 0 and size_guess > MAX_CONTENT_BYTES * 2:
            pages.append({
                "url": url, "final_url": url, "status": st_head or 200,
                "redirect_chain": [], "issues": ["skipped_large_content"],
                "title": None, "meta_description": None, "h1": None,
                "h2": [], "h3": [], "canonical": None,
                "noindex": None, "nofollow": None
            })
            continue

        status, final_url, html, rchain = _get_with_cap(url, ua)
        if rchain:
            issues.append("redirect")

        title = description = canonical = None
        h1 = None; h2: List[str] = []; h3: List[str] = []
        noindex = None; nofollow = None

        if status == 200 and html:
            soup = BeautifulSoup(html, "html.parser")
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            mdesc = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
            if mdesc and mdesc.get("content"):
                description = mdesc["content"].strip()
            link_tag = soup.find("link", rel=re.compile("canonical", re.I))
            if link_tag and link_tag.get("href"):
                canonical = urljoin(final_url, link_tag.get("href"))
            noindex, nofollow = _extract_meta_robots(soup)
            h1_tag = soup.find("h1")
            if h1_tag:
                h1 = h1_tag.get_text(strip=True)
            for tag in soup.find_all("h2"): h2.append(tag.get_text(strip=True))
            for tag in soup.find_all("h3"): h3.append(tag.get_text(strip=True))

            # ontdek interne links (bounded queue)
            if len(q) < MAX_QUEUE_SIZE:
                for a in soup.find_all("a", href=True):
                    absu = urljoin(final_url, a["href"])
                    if _same_site(final_url, absu) and absu not in visited:
                        q.append(absu)

        # issues obv status/inhoud
        if status == 200:
            if not title: issues.append("missing_title")
            if not h1: issues.append("missing_h1")
            if not description: issues.append("missing_meta_description")
        elif 300 <= status < 400:
            issues.append("redirect_http")
        elif status == 310:
            issues.append("too_many_redirects")
        elif status == 0:
            issues.append("fetch_error")
        elif status >= 400:
            issues.append(f"http_{status}")

        pages.append({
            "url": url,
            "final_url": final_url or url,
            "status": status,
            "redirect_chain": rchain[:MAX_REDIRECTS],
            "title": title,
            "meta_description": description,
            "h1": h1,
            "h2": h2,
            "h3": h3,
            "canonical": canonical,
            "noindex": noindex,
            "nofollow": nofollow,
            "issues": issues
        })

    dur_ms = int((time.time() - t0) * 1000)
    summary = {
        "pages_total": len(pages),
        "ok_200": sum(1 for p in pages if isinstance(p.get("status"), int) and p["status"] == 200),
        "redirect_3xx": sum(1 for p in pages if isinstance(p.get("status"), int) and 300 <= p["status"] < 400),
        "errors_4xx_5xx": sum(1 for p in pages if isinstance(p.get("status"), int) and p["status"] >= 400),
        "fetch_errors": sum(1 for p in pages if p.get("status") == 0),
        "duration_ms": dur_ms,
        "capped_by_runtime": len(visited) >= max_pages
    }

    # quick wins
    quick_wins = []
    missing_title = [p["final_url"] for p in pages if p.get("status") == 200 and not p.get("title")]
    if missing_title: quick_wins.append({"type": "missing_title", "count": len(missing_title), "examples": missing_title[:5]})
    missing_h1 = [p["final_url"] for p in pages if p.get("status") == 200 and not p.get("h1")]
    if missing_h1: quick_wins.append({"type": "missing_h1", "count": len(missing_h1), "examples": missing_h1[:5]})
    missing_desc = [p["final_url"] for p in pages if p.get("status") == 200 and not p.get("meta_description")]
    if missing_desc: quick_wins.append({"type": "missing_meta_description", "count": len(missing_desc), "examples": missing_desc[:5]})
    redirects = [p for p in pages if "redirect" in (p.get("issues") or [])]
    if redirects: quick_wins.append({"type": "redirects", "count": len(redirects)})

    return {
        "start_url": start_url,
        "pages": pages,
        "summary": summary,
        "quick_wins": quick_wins
    }
