# crawl_light.py
# V13: ultra-low memory crawler using streaming + selectolax (fast & light)
import os
import re
import time
import requests
from urllib.parse import urljoin, urlparse
from typing import Dict, Any, List, Set, Tuple
from collections import deque
from selectolax.parser import HTMLParser

DEFAULT_UA = "AseonBot/0.3 (+https://aseon.ai)"
MAX_REDIRECTS = int(os.getenv("CRAWL_MAX_REDIRECTS", "4"))
REQ_TIMEOUT = int(os.getenv("CRAWL_REQ_TIMEOUT_SEC", "8"))
MAX_QUEUE_SIZE = int(os.getenv("CRAWL_MAX_QUEUE", "400"))
MAX_CONTENT_BYTES = int(os.getenv("CRAWL_CONTENT_CAP_BYTES", "350000"))   # ~0.35 MB per page

def _registrable_host(netloc: str) -> str:
    n = (netloc or "").lower()
    return n[4:] if n.startswith("www.") else n

def _same_site(u1: str, u2: str) -> bool:
    a = urlparse(u1); b = urlparse(u2)
    return _registrable_host(a.netloc) == _registrable_host(b.netloc)

def _allowed_by_robots(robots_txt: str, path: str) -> bool:
    consider = False
    for line in (robots_txt or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        if line.lower().startswith("user-agent:"):
            ua = line.split(":", 1)[1].strip().lower()
            consider = (ua in ("*", "aseonbot", "aseonbot/0.3 (+https://aseon.ai)"))
        elif consider and line.lower().startswith("disallow:"):
            rule = line.split(":", 1)[1].strip()
            rel = path or "/"
            if rule and rel.startswith(rule):
                return False
    return True

def _load_robots(start_url: str, ua: str) -> str:
    pu = urlparse(start_url)
    host = _registrable_host(pu.netloc)
    candidates = [
        f"{pu.scheme}://{pu.netloc}/robots.txt",
        f"{pu.scheme}://www.{host}/robots.txt",
        f"{pu.scheme}://{host}/robots.txt",
    ]
    for u in candidates:
        try:
            r = requests.get(u, headers={"User-Agent": ua}, timeout=REQ_TIMEOUT)
            try:
                if r.status_code == 200 and len(r.text) < 200_000:
                    return r.text
            finally:
                r.close()
        except Exception:
            pass
    return ""

def _get_capped(url: str, ua: str) -> Tuple[int, str, str, List[str]]:
    """
    GET with redirects, stream body, cap to MAX_CONTENT_BYTES.
    Returns (status, final_url, html_text (capped), redirect_chain)
    """
    chain: List[str] = []
    with requests.Session() as s:
        s.max_redirects = MAX_REDIRECTS
        try:
            resp = s.get(url, headers={"User-Agent": ua}, timeout=REQ_TIMEOUT, allow_redirects=True, stream=True)
        except requests.exceptions.TooManyRedirects:
            return 310, url, "", ["too_many_redirects"]
        except Exception:
            return 0, url, "", []
        try:
            status = resp.status_code
            final_url = resp.url
            for h in resp.history[:MAX_REDIRECTS]:
                chain.append(h.headers.get("Location") or h.url)

            if status != 200:
                return status, final_url, "", chain

            # Stream bytes into a small buffer
            buf = bytearray()
            for chunk in resp.iter_content(chunk_size=24_576):
                if not chunk:
                    break
                if len(buf) + len(chunk) > MAX_CONTENT_BYTES:
                    # cap reached
                    break
                buf.extend(chunk)
            html = buf.decode(resp.encoding or "utf-8", errors="replace")
            return status, final_url, html, chain
        finally:
            resp.close()

def _extract_meta_robots(tree: HTMLParser) -> Tuple[bool, bool]:
    noindex = nofollow = False
    for n in tree.css('meta[name="robots"]'):
        c = (n.attributes.get("content") or "").lower()
        if "noindex" in c: noindex = True
        if "nofollow" in c: nofollow = True
    return noindex, nofollow

def _parse_page(final_url: str, html: str) -> Dict[str, Any]:
    if not html:
        return {"title": None, "meta_description": None, "h1": None, "h2": [], "h3": [], "canonical": None, "noindex": None, "nofollow": None}
    tree = HTMLParser(html)

    # title
    title_el = tree.css_first("title")
    title = title_el.text().strip() if title_el else None

    # meta description
    desc_el = tree.css_first('meta[name="description"]')
    description = (desc_el.attributes.get("content").strip() if desc_el and desc_el.attributes.get("content") else None)

    # canonical
    can_el = tree.css_first('link[rel="canonical"]')
    canonical = urljoin(final_url, can_el.attributes.get("href")) if can_el and can_el.attributes.get("href") else None

    # robots
    noindex, nofollow = _extract_meta_robots(tree)

    # headings
    h1_el = tree.css_first("h1")
    h1 = h1_el.text().strip() if h1_el else None
    h2 = [n.text().strip() for n in tree.css("h2")[:20]]
    h3 = [n.text().strip() for n in tree.css("h3")[:30]]

    # internal links (bounded elsewhere)
    links = []
    for a in tree.css("a[href]")[:400]:
        href = a.attributes.get("href")
        if href:
            links.append(href)

    return {
        "title": title, "meta_description": description, "h1": h1, "h2": h2, "h3": h3,
        "canonical": canonical, "noindex": noindex, "nofollow": nofollow,
        "_links": links
    }

def crawl_site(start_url: str, max_pages: int = 12, ua: str = DEFAULT_UA) -> Dict[str, Any]:
    # normalize
    start_url = start_url.strip()
    if not start_url.startswith(("http://", "https://")):
        start_url = "https://" + start_url
    if not start_url.endswith("/"): start_url += "/"

    robots_txt = _load_robots(start_url, ua)

    q: deque[str] = deque([start_url])
    visited: Set[str] = set()
    pages: List[Dict[str, Any]] = []
    t0 = time.time()

    count = 0
    while q and count < max_pages:
        url = q.popleft()
        if url in visited:
            continue
        visited.add(url)
        count += 1

        pu = urlparse(url)
        if not _allowed_by_robots(robots_txt, pu.path):
            pages.append({"url": url, "final_url": url, "status": 999, "redirect_chain": [], "issues": ["blocked_by_robots"],
                          "title": None, "meta_description": None, "h1": None, "h2": [], "h3": [], "canonical": None,
                          "noindex": None, "nofollow": None})
            continue

        status, final_url, html, rchain = _get_capped(url, ua)
        issues: List[str] = []
        if rchain: issues.append("redirect")

        parsed = _parse_page(final_url, html) if status == 200 else {
            "title": None, "meta_description": None, "h1": None, "h2": [], "h3": [], "canonical": None, "noindex": None, "nofollow": None, "_links": []
        }

        # discover internal links (bounded queue)
        for href in (parsed.get("_links") or []):
            absu = urljoin(final_url, href)
            if _same_site(final_url, absu) and absu not in visited and len(q) < MAX_QUEUE_SIZE:
                q.append(absu)

        # quality issues
        if status == 200:
            if not parsed["title"]: issues.append("missing_title")
            if not parsed["h1"]: issues.append("missing_h1")
            if not parsed["meta_description"]: issues.append("missing_meta_description")
        elif 300 <= status < 400:
            issues.append("redirect_http")
        elif status == 310:
            issues.append("too_many_redirects")
        elif status == 0:
            issues.append("fetch_error")
        elif status >= 400:
            issues.append(f"http_{status}")

        pages.append({
            "url": url, "final_url": final_url, "status": status, "redirect_chain": rchain[:MAX_REDIRECTS],
            "title": parsed["title"], "meta_description": parsed["meta_description"], "h1": parsed["h1"],
            "h2": parsed["h2"], "h3": parsed["h3"], "canonical": parsed["canonical"],
            "noindex": parsed["noindex"], "nofollow": parsed["nofollow"], "issues": issues
        })

        # free memory aggressively
        del html, parsed

    dur_ms = int((time.time() - t0) * 1000)
    summary = {
        "pages_total": len(pages),
        "ok_200": sum(1 for p in pages if p.get("status") == 200),
        "redirect_3xx": sum(1 for p in pages if isinstance(p.get("status"), int) and 300 <= p["status"] < 400),
        "errors_4xx_5xx": sum(1 for p in pages if isinstance(p.get("status"), int) and p["status"] >= 400),
        "duration_ms": dur_ms,
        "capped_by_runtime": len(visited) >= max_pages
    }

    # quick wins (compact)
    quick_wins = []
    if any(p.get("status") == 200 and not p.get("title") for p in pages):
        quick_wins.append({"type": "missing_title"})
    if any(p.get("status") == 200 and not p.get("h1") for p in pages):
        quick_wins.append({"type": "missing_h1"})
    if any(p.get("status") == 200 and not p.get("meta_description") for p in pages):
        quick_wins.append({"type": "missing_meta_description"})
    if any("redirect" in (p.get("issues") or []) for p in pages):
        quick_wins.append({"type": "redirects"})

    return {"start_url": start_url, "pages": pages, "summary": summary, "quick_wins": quick_wins}
