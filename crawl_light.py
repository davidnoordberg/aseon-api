# crawl_light.py
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import Dict, Any, List, Set, Tuple
from collections import deque

DEFAULT_UA = "AseonBot/0.2 (+https://aseon.ai)"

def _fetch(url: str, ua: str, timeout: int = 12, allow_redirects: bool = False):
    return requests.get(url, headers={"User-Agent": ua}, timeout=timeout, allow_redirects=allow_redirects)

def _allowed_by_robots(base: str, path: str, robots_txt: str) -> bool:
    # ultra-light robots parser (Disallow only)
    allow = True
    rules = []
    agent = None
    for line in robots_txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"): 
            continue
        if line.lower().startswith("user-agent:"):
            agent = line.split(":",1)[1].strip().lower()
        elif line.lower().startswith("disallow:") and (agent in ("*", "aseonbot/0.2 (+https://aseon.ai)","aseonbot", None)):
            rule = line.split(":",1)[1].strip()
            rules.append(rule)
    rel = path or "/"
    for rule in rules:
        if rule and rel.startswith(rule):
            return False
    return allow

def _extract_meta_robots(soup: BeautifulSoup) -> Tuple[bool, bool]:
    noindex = False
    nofollow = False
    tag = soup.find("meta", attrs={"name": re.compile("^robots$", re.I)})
    if tag and tag.get("content"):
        content = tag["content"].lower()
        if "noindex" in content: noindex = True
        if "nofollow" in content: nofollow = True
    return noindex, nofollow

def _abs_base(url: str) -> str:
    u = urlparse(url)
    return f"{u.scheme}://{u.netloc}"

def crawl_site(start_url: str, max_pages: int = 20, ua: str = DEFAULT_UA) -> Dict[str, Any]:
    """
    Lightweight BFS crawler with essentials:
    - Respects robots.txt Disallow
    - Captures: status, redirects (Location), title, meta description, H1–H3, canonical, noindex/nofollow
    - Collects internal links (same host) up to max_pages
    - Emits issues[] and quick_wins[]
    """
    start_url = start_url.rstrip("/") + "/"
    base = _abs_base(start_url)
    visited: Set[str] = set()
    q: deque[str] = deque([start_url])
    pages: List[Dict[str, Any]] = []
    t0 = time.time()

    # robots
    robots_txt = ""
    try:
        r_robots = _fetch(urljoin(base, "/robots.txt"), ua, timeout=8, allow_redirects=True)
        if r_robots.status_code == 200:
            robots_txt = r_robots.text
    except Exception:
        robots_txt = ""

    count = 0
    while q and count < max_pages:
        url = q.popleft()
        if url in visited:
            continue
        visited.add(url)
        count += 1

        issues: List[str] = []
        try:
            r = _fetch(url, ua, allow_redirects=False)
            status = r.status_code
            final_url = url
            location = r.headers.get("Location")
            if status in (301,302,303,307,308) and location:
                pages.append({
                    "url": url, "status": status, "redirect_to": urljoin(url, location),
                    "issues": ["redirect"], "noindex": None, "nofollow": None
                })
                continue

            html = r.text if status == 200 else ""
        except Exception as e:
            pages.append({"url": url, "status": "error", "error": str(e), "issues": ["fetch_error"]})
            continue

        soup = BeautifulSoup(html, "html.parser") if html else None
        title = description = canonical = None
        h1 = None; h2: List[str] = []; h3: List[str] = []
        noindex = None; nofollow = None

        # robots per-path
        if not _allowed_by_robots(base, urlparse(url).path, robots_txt):
            issues.append("blocked_by_robots")

        if soup:
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            mdesc = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
            if mdesc and mdesc.get("content"):
                description = mdesc["content"].strip()
            link_tag = soup.find("link", rel=re.compile("canonical", re.I))
            if link_tag and link_tag.get("href"):
                canonical = urljoin(base, link_tag.get("href"))

            noindex, nofollow = _extract_meta_robots(soup)

            h1_tag = soup.find("h1")
            if h1_tag:
                h1 = h1_tag.get_text(strip=True)
            for tag in soup.find_all("h2"): h2.append(tag.get_text(strip=True))
            for tag in soup.find_all("h3"): h3.append(tag.get_text(strip=True))

            # discover internal links
            for a in soup.find_all("a", href=True):
                absu = urljoin(url, a["href"])
                if absu.startswith(base) and absu not in visited:
                    q.append(absu)

        # issues
        if status == 200:
            if not title: issues.append("missing_title")
            if not h1: issues.append("missing_h1")
            if not description: issues.append("missing_meta_description")
        elif status >= 400:
            issues.append(f"http_{status}")

        pages.append({
            "url": url,
            "status": status,
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

    # quick wins (simple heuristics)
    quick_wins = []
    missing_title = [p["url"] for p in pages if isinstance(p.get("status"), int) and p["status"] == 200 and not p.get("title")]
    if missing_title: quick_wins.append({"type": "missing_title", "count": len(missing_title), "examples": missing_title[:5]})
    missing_h1 = [p["url"] for p in pages if isinstance(p.get("status"), int) and p["status"] == 200 and not p.get("h1")]
    if missing_h1: quick_wins.append({"type": "missing_h1", "count": len(missing_h1), "examples": missing_h1[:5]})
    missing_desc = [p["url"] for p in pages if isinstance(p.get("status"), int) and p["status"] == 200 and not p.get("meta_description")]
    if missing_desc: quick_wins.append({"type": "missing_meta_description", "count": len(missing_desc), "examples": missing_desc[:5]})
    redirects = [p for p in pages if isinstance(p.get("status"), int) and 300 <= p["status"] < 400]
    if redirects: quick_wins.append({"type": "redirects", "count": len(redirects)})
    errors = [p for p in pages if isinstance(p.get("status"), int) and p["status"] >= 400]
    if errors: quick_wins.append({"type": "http_errors", "count": len(errors)})

    return {
        "start_url": start_url,
        "pages": pages,
        "summary": summary,
        "quick_wins": quick_wins
    }
