# crawl_light.py
from __future__ import annotations
import re, time
from urllib.parse import urljoin, urlparse, urldefrag
import requests
from bs4 import BeautifulSoup
from urllib import robotparser

DEFAULT_UA = "AseonBot/0.1 (+https://aseon.ai)"
FETCH_TIMEOUT = 10  # sec
MAX_BODY_BYTES = 2_000_000  # 2 MB safeguard

def normalize_start_url(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    if not url.endswith("/"):
        url += "/"
    return url

def same_host(a: str, b: str) -> bool:
    return urlparse(a).netloc.lower() == urlparse(b).netloc.lower()

def can_fetch(robots_url: str, ua: str, url: str) -> bool:
    rp = robotparser.RobotFileParser()
    try:
        resp = requests.get(robots_url, headers={"User-Agent": ua}, timeout=FETCH_TIMEOUT)
        if resp.status_code >= 400:
            return True  # geen robots → ga door
        rp.parse(resp.text.splitlines())
        return rp.can_fetch(ua, url)
    except requests.RequestException:
        return True

def fetch(url: str, ua: str) -> tuple[int, str, dict]:
    try:
        r = requests.get(url, headers={"User-Agent": ua}, timeout=FETCH_TIMEOUT, allow_redirects=True)
        body = r.text[:MAX_BODY_BYTES] if "text/html" in r.headers.get("Content-Type","").lower() else ""
        return r.status_code, body, dict(r.headers)
    except requests.RequestException:
        return 0, "", {}

def extract(html: str) -> dict:
    if not html:
        return {"title": None, "description": None, "canonical": None, "h1": None, "h2": [], "h3": []}
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title else None
    desc = None
    md = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    if md and md.get("content"):
        desc = md["content"].strip()
    canonical = None
    link_canon = soup.find("link", rel=lambda v: v and "canonical" in v)
    if link_canon and link_canon.get("href"):
        canonical = link_canon["href"].strip()
    get_text = lambda el: re.sub(r"\s+", " ", el.get_text(strip=True)) if el else None
    h1 = get_text(soup.find("h1"))
    h2 = [get_text(h) for h in soup.find_all("h2")][:5]
    h3 = [get_text(h) for h in soup.find_all("h3")][:5]
    return {"title": title, "description": desc, "canonical": canonical, "h1": h1, "h2": h2, "h3": h3}

def discover_links(base_url: str, html: str) -> list[str]:
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        absu = urljoin(base_url, href)
        absu, _ = urldefrag(absu)
        links.append(absu)
    # dedup maar behoud volgorde
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def crawl_site(start_url: str, max_pages: int = 10, ua: str = DEFAULT_UA) -> dict:
    start_url = normalize_start_url(start_url)
    robots_url = urljoin(start_url, "/robots.txt")
    results = []
    q = [start_url]
    seen = set()
    t0 = time.time()

    while q and len(results) < max_pages:
        url = q.pop(0)
        if url in seen:
            continue
        seen.add(url)

        if not same_host(start_url, url):
            continue
        if not can_fetch(robots_url, ua, url):
            results.append({"url": url, "status": 999, "skipped": "robots", "title": None, "h1": None, "issues": ["blocked_by_robots"]})
            continue

        status, html, headers = fetch(url, ua)
        meta = extract(html)
        issues = []
        if status == 0:
            issues.append("fetch_error")
        if status == 404:
            issues.append("not_found")
        if not meta.get("title"):
            issues.append("missing_title")
        if not meta.get("h1"):
            issues.append("missing_h1")

        results.append({
            "url": url,
            "status": status,
            "title": meta.get("title"),
            "description": meta.get("description"),
            "canonical": meta.get("canonical"),
            "h1": meta.get("h1"),
            "h2": meta.get("h2"),
            "h3": meta.get("h3"),
            "issues": issues
        })

        if status == 200 and html:
            for link in discover_links(url, html):
                if same_host(start_url, link) and link not in seen and len(q) + len(results) < max_pages:
                    q.append(link)

    summary = {
        "pages_total": len(results),
        "ok_200": sum(1 for r in results if r["status"] == 200),
        "not_found_404": sum(1 for r in results if r["status"] == 404),
        "redirect_3xx": sum(1 for r in results if 300 <= (r["status"] or 0) < 400),
        "fetch_errors": sum(1 for r in results if r["status"] == 0),
        "duration_ms": int((time.time() - t0) * 1000),
    }
    quick_wins = []
    for r in results:
        if "missing_title" in r["issues"]:
            quick_wins.append({"url": r["url"], "win": "Voeg een duidelijke <title> toe"})
        if "missing_h1" in r["issues"]:
            quick_wins.append({"url": r["url"], "win": "Voeg een H1-kop toe"})
        if r["status"] == 404:
            quick_wins.append({"url": r["url"], "win": "Fix 404 of maak redirect"})

    return {"start_url": start_url, "summary": summary, "pages": results, "quick_wins": quick_wins}
