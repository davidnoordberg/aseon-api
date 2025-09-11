# crawl_light.py (scaled to ~1000 pages safely)
from __future__ import annotations
import re, time, xml.etree.ElementTree as ET
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag
from urllib import robotparser

import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup

DEFAULT_UA = "AseonBot/0.2 (+https://aseon.ai)"
FETCH_TIMEOUT = 10          # sec
MAX_BODY_BYTES = 2_000_000  # 2 MB cap
SITEMAP_MAX_URLS = 5000     # hard cap reading sitemaps

def _normalize_start_url(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    if not url.endswith("/"):
        url += "/"
    return url

def _same_host(a: str, b: str) -> bool:
    return urlparse(a).netloc.lower() == urlparse(b).netloc.lower()

def _session(ua: str) -> requests.Session:
    s = requests.Session()
    retries = Retry(total=2, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("http://", HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20))
    s.mount("https://", HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20))
    s.headers.update({"User-Agent": ua, "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8"})
    return s

def _can_fetch(rp: robotparser.RobotFileParser, ua: str, url: str) -> bool:
    try:
        return rp.can_fetch(ua, url)
    except Exception:
        return True

def _load_robots_and_sitemaps(sess: requests.Session, start_url: str, ua: str):
    robots_url = urljoin(start_url, "/robots.txt")
    rp = robotparser.RobotFileParser()
    try:
        r = sess.get(robots_url, timeout=FETCH_TIMEOUT)
        if r.status_code < 400:
            rp.parse(r.text.splitlines())
        else:
            rp.parse([])  # treat as empty
    except requests.RequestException:
        rp.parse([])

    sitemap_urls = []
    try:
        # robots.txt may contain Sitemap: entries
        r = sess.get(robots_url, timeout=FETCH_TIMEOUT)
        if r.ok:
            for line in r.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    sm = line.split(":", 1)[1].strip()
                    if sm:
                        sitemap_urls.append(sm)
    except requests.RequestException:
        pass

    # also try /sitemap.xml if none found
    if not sitemap_urls:
        sitemap_urls = [urljoin(start_url, "/sitemap.xml")]

    return rp, sitemap_urls

def _fetch(sess: requests.Session, url: str) -> tuple[int, str, dict]:
    try:
        r = sess.get(url, timeout=FETCH_TIMEOUT, allow_redirects=True)
        ctype = r.headers.get("Content-Type", "").lower()
        body = r.text[:MAX_BODY_BYTES] if "text/html" in ctype or "xml" in ctype else ""
        return r.status_code, body, dict(r.headers)
    except requests.RequestException:
        return 0, "", {}

def _extract(html: str) -> dict:
    if not html:
        return {"title": None, "description": None, "canonical": None, "h1": None, "h2": [], "h3": [], "noindex": False, "nofollow": False}
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

    # robots meta
    noindex = False
    nofollow = False
    for m in soup.find_all("meta", attrs={"name": re.compile(r"^robots$", re.I)}):
        content = (m.get("content") or "").lower()
        if "noindex" in content: noindex = True
        if "nofollow" in content: nofollow = True

    # headings
    get_text = lambda el: re.sub(r"\s+", " ", el.get_text(strip=True)) if el else None
    h1 = get_text(soup.find("h1"))
    h2 = [get_text(h) for h in soup.find_all("h2")][:5]
    h3 = [get_text(h) for h in soup.find_all("h3")][:5]

    return {"title": title, "description": desc, "canonical": canonical, "h1": h1, "h2": h2, "h3": h3, "noindex": noindex, "nofollow": nofollow}

def _discover_links(base_url: str, html: str) -> list[str]:
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
    # dedup keep order
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def _sitemap_seed(sess: requests.Session, start_url: str, sitemap_urls: list[str], max_seed: int) -> list[str]:
    seeds = []
    for sm in sitemap_urls:
        try:
            r = sess.get(sm, timeout=FETCH_TIMEOUT)
            if not r.ok:
                continue
            # basic sitemap parsing (xml)
            try:
                root = ET.fromstring(r.text)
            except ET.ParseError:
                continue
            ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            for loc in root.findall(".//sm:loc", ns):
                url = (loc.text or "").strip()
                if url:
                    seeds.append(url)
                    if len(seeds) >= min(SITEMAP_MAX_URLS, max_seed):
                        return seeds
        except requests.RequestException:
            continue
    return seeds[:max_seed]

def crawl_site(
    start_url: str,
    max_pages: int = 100,
    ua: str = DEFAULT_UA,
    max_depth: int = 5,
    delay_ms: int = 150,
    max_runtime_s: int = 180,
) -> dict:
    """
    Kleine, veilige crawler met sitemap-seed en grenzen.
    """
    t0 = time.time()
    start_url = _normalize_start_url(start_url)
    sess = _session(ua)

    rp, sitemap_urls = _load_robots_and_sitemaps(sess, start_url, ua)

    # queue (url, depth)
    q: deque[tuple[str, int]] = deque()
    seen: set[str] = set()
    results = []

    # seed: homepage
    q.append((start_url, 0))

    # seed uit sitemap (tot ~max_pages, maar niet te veel)
    sm_seeds = _sitemap_seed(sess, start_url, sitemap_urls, max_seed=max_pages)
    for u in sm_seeds:
        if _same_host(start_url, u):
            q.append((u, 0))

    while q and len(results) < max_pages and (time.time() - t0) < max_runtime_s:
        url, depth = q.popleft()
        if url in seen:
            continue
        seen.add(url)

        if not _same_host(start_url, url):
            continue
        if not _can_fetch(rp, ua, url):
            results.append({"url": url, "status": 999, "skipped": "robots", "title": None, "h1": None, "issues": ["blocked_by_robots"]})
            continue

        status, html, headers = _fetch(sess, url)
        meta = _extract(html)

        issues = []
        if status == 0:
            issues.append("fetch_error")
        if status == 404:
            issues.append("not_found")
        if not meta.get("title"):
            issues.append("missing_title")
        if not meta.get("h1"):
            issues.append("missing_h1")
        if meta.get("noindex"):
            issues.append("noindex")
        if meta.get("nofollow"):
            issues.append("nofollow")

        results.append({
            "url": url,
            "status": status,
            "title": meta.get("title"),
            "description": meta.get("description"),
            "canonical": meta.get("canonical"),
            "h1": meta.get("h1"),
            "h2": meta.get("h2"),
            "h3": meta.get("h3"),
            "noindex": meta.get("noindex"),
            "nofollow": meta.get("nofollow"),
            "issues": issues
        })

        if status == 200 and html and depth < max_depth:
            for link in _discover_links(url, html):
                if _same_host(start_url, link) and link not in seen and len(results) + len(q) < max_pages:
                    q.append((link, depth + 1))

        # politeness + hard runtime guard
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        if (time.time() - t0) >= max_runtime_s:
            break

    summary = {
        "pages_total": len(results),
        "ok_200": sum(1 for r in results if r["status"] == 200),
        "not_found_404": sum(1 for r in results if r["status"] == 404),
        "redirect_3xx": sum(1 for r in results if 300 <= (r["status"] or 0) < 400),
        "fetch_errors": sum(1 for r in results if r["status"] == 0),
        "duration_ms": int((time.time() - t0) * 1000),
        "capped_by_runtime": (time.time() - t0) >= max_runtime_s,
    }

    quick_wins = []
    for r in results:
        if "missing_title" in r["issues"]:
            quick_wins.append({"url": r["url"], "win": "Voeg een duidelijke <title> toe"})
        if "missing_h1" in r["issues"]:
            quick_wins.append({"url": r["url"], "win": "Voeg een H1-kop toe"})
        if r["status"] == 404:
            quick_wins.append({"url": r["url"], "win": "Fix 404 of maak redirect"})
        if "noindex" in r["issues"]:
            quick_wins.append({"url": r["url"], "win": "Pagina staat op noindex (check bewust?)"})

    return {
        "start_url": start_url,
        "summary": summary,
        "pages": results,
        "quick_wins": quick_wins
    }
