# crawl_light.py
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import Dict, Any, List, Set, Tuple
from collections import deque

DEFAULT_UA = "AseonBot/0.2 (+https://aseon.ai)"

def _registrable_host(netloc: str) -> str:
    """
    Heuristiek zonder tldextract:
    - lowercased
    - strip 'www.' prefix
    Dit dekt www <-> non-www gevallen voor de meeste sites.
    """
    n = (netloc or "").lower()
    if n.startswith("www."):
        n = n[4:]
    return n

def _same_site(u1: str, u2: str) -> bool:
    a = urlparse(u1); b = urlparse(u2)
    return _registrable_host(a.netloc) == _registrable_host(b.netloc)

def _fetch(url: str, ua: str, timeout: int = 12, allow_redirects: bool = True):
    return requests.get(
        url,
        headers={"User-Agent": ua},
        timeout=timeout,
        allow_redirects=allow_redirects,
    )

def _allowed_by_robots(base: str, path: str, robots_txt: str) -> bool:
    # ultra-light robots parser (Disallow only; applies to * and our UA)
    allow = True
    consider = False
    for line in robots_txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("user-agent:"):
            ua = line.split(":", 1)[1].strip().lower()
            consider = (ua in ("*", "aseonbot/0.2 (+https://aseon.ai)", "aseonbot"))
        elif consider and line.lower().startswith("disallow:"):
            rule = line.split(":", 1)[1].strip()
            rel = path or "/"
            if rule and rel.startswith(rule):
                return False
    return allow

def _extract_meta_robots(soup: BeautifulSoup) -> Tuple[bool, bool]:
    noindex = nofollow = False
    tag = soup.find("meta", attrs={"name": re.compile("^robots$", re.I)})
    if tag and tag.get("content"):
        content = tag["content"].lower()
        if "noindex" in content: noindex = True
        if "nofollow" in content: nofollow = True
    return noindex, nofollow

def crawl_site(start_url: str, max_pages: int = 20, ua: str = DEFAULT_UA) -> Dict[str, Any]:
    """
    Lightweight BFS crawler (redirect-aware):
    - Volgt redirects en parseert content op de eind-URL.
    - Behandelt www en non-www als dezelfde site.
    - Respecteert robots.txt Disallow (basic).
    - Houdt bij: status, redirect_chain, title, meta description, H1–H3, canonical, noindex/nofollow, issues.
    - Vindt interne links (zelfde registrable host) en crawlt door tot max_pages.
    """
    # normaliseer start_url
    start_url = start_url.strip()
    if not start_url.startswith(("http://", "https://")):
        start_url = "https://" + start_url
    if not start_url.endswith("/"):
        start_url += "/"

    visited: Set[str] = set()
    q: deque[str] = deque([start_url])
    pages: List[Dict[str, Any]] = []
    t0 = time.time()

    # robots ophalen op basis van registrable host
    start_parsed = urlparse(start_url)
    robots_host = _registrable_host(start_parsed.netloc)
    robots_base = f"{start_parsed.scheme}://{start_parsed.netloc}"
    # probeer ook met www. en zonder www. als fallback
    robots_urls = [
        f"{start_parsed.scheme}://{start_parsed.netloc}/robots.txt",
        f"{start_parsed.scheme}://www.{robots_host}/robots.txt",
        f"{start_parsed.scheme}://{robots_host}/robots.txt",
    ]
    robots_txt = ""
    for ru in robots_urls:
        try:
            r_robots = _fetch(ru, ua, timeout=6, allow_redirects=True)
            if r_robots.status_code == 200 and len(r_robots.text) < 200_000:
                robots_txt = r_robots.text
                break
        except Exception:
            pass

    count = 0
    while q and count < max_pages:
        url = q.popleft()
        if url in visited:
            continue
        visited.add(url)
        count += 1

        issues: List[str] = []
        redirect_chain: List[str] = []

        # robots check
        try:
            if not _allowed_by_robots(robots_base, urlparse(url).path, robots_txt):
                pages.append({
                    "url": url, "status": 999, "issues": ["blocked_by_robots"],
                    "noindex": None, "nofollow": None
                })
                continue
        except Exception:
            # als robots faalt, ga gewoon verder
            pass

        # fetch met redirects AAN; we registreren history
        try:
            r = _fetch(url, ua, allow_redirects=True)
            status = r.status_code
            final_url = r.url
            if r.history:
                redirect_chain = [h.headers.get("Location", "") or h.url for h in r.history]
                issues.append("redirect")
        except Exception as e:
            pages.append({"url": url, "status": "error", "error": str(e), "issues": ["fetch_error"]})
            continue

        html = r.text if status == 200 else ""
        soup = BeautifulSoup(html, "html.parser") if html else None

        title = description = canonical = None
        h1 = None; h2: List[str] = []; h3: List[str] = []
        noindex = None; nofollow = None

        if soup:
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

            # discover internal links on the final_url host (registrable host match)
            for a in soup.find_all("a", href=True):
                absu = urljoin(final_url, a["href"])
                if _same_site(final_url, absu) and absu not in visited:
                    q.append(absu)

        # issues by status
        if status == 200:
            if not title: issues.append("missing_title")
            if not h1: issues.append("missing_h1")
            if not description: issues.append("missing_meta_description")
        elif 300 <= status < 400:
            issues.append("redirect_http")
        elif status >= 400:
            issues.append(f"http_{status}")

        pages.append({
            "url": url,
            "final_url": r.url,
            "status": status,
            "redirect_chain": redirect_chain,
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
        "fetch_errors": sum(1 for p in pages if p.get("status") == "error"),
        "duration_ms": dur_ms,
        "capped_by_runtime": len(visited) >= max_pages
    }

    # quick wins
    quick_wins = []
    missing_title = [p["final_url"] for p in pages if isinstance(p.get("status"), int) and p["status"] == 200 and not p.get("title")]
    if missing_title: quick_wins.append({"type": "missing_title", "count": len(missing_title), "examples": missing_title[:5]})
    missing_h1 = [p["final_url"] for p in pages if isinstance(p.get("status"), int) and p["status"] == 200 and not p.get("h1")]
    if missing_h1: quick_wins.append({"type": "missing_h1", "count": len(missing_h1), "examples": missing_h1[:5]})
    missing_desc = [p["final_url"] for p in pages if isinstance(p.get("status"), int) and p["status"] == 200 and not p.get("meta_description")]
    if missing_desc: quick_wins.append({"type": "missing_meta_description", "count": len(missing_desc), "examples": missing_desc[:5]})
    redirects = [p for p in pages if ("redirect" in (p.get("issues") or []))]
    if redirects: quick_wins.append({"type": "redirects", "count": len(redirects)})

    return {
        "start_url": start_url,
        "pages": pages,
        "summary": summary,
        "quick_wins": quick_wins
    }
