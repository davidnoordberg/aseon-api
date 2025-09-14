# crawl_light.py (UTF-8 safe + HTML-only + asset-filter)
import re, time, gc
from urllib.parse import urljoin, urlparse
import requests
import os

DEFAULT_TIMEOUT = 10
MAX_HTML_BYTES = int(os.getenv("CRAWL_MAX_HTML_BYTES", "500000"))  # 500 KB cap
HEADERS_TEMPLATE = lambda ua: {"User-Agent": ua or "AseonBot/0.2 (+https://aseon.ai)"}

# Skip duidelijk non-HTML
_SKIP_EXT = {
    ".png",".jpg",".jpeg",".webp",".gif",".svg",".ico",".bmp",".avif",
    ".pdf",".zip",".rar",".7z",".gz",".mp4",".mp3",".mov",".wav",".woff",".woff2",".ttf",".eot"
}

def _seems_asset(url: str) -> bool:
    path = urlparse(url).path.lower()
    if any(path.endswith(ext) for ext in _SKIP_EXT):
        return True
    # veelvoorkomende junk
    if "/wp-json" in path or "?" in path and ("utm_" in url or "fbclid=" in url):
        return False  # niet per se asset
    # favicons / icons
    if "favicon" in path or "apple-touch-icon" in path:
        return True
    return False

def _normalize_host(u: str) -> str:
    try:
        p = urlparse(u)
        host = p.netloc.lower()
        if host.startswith("www."): host = host[4:]
        return host
    except Exception:
        return u

def _fetch(url: str, ua: str):
    resp = requests.get(
        url,
        headers=HEADERS_TEMPLATE(ua),
        timeout=DEFAULT_TIMEOUT,
        allow_redirects=True
    )
    final_url = str(resp.url)

    # Kies een goede encoding vóór we .text nemen
    # Prefer server-declared; als het ISO-8859-1 / None is → apparent_encoding → fallback utf-8
    enc = (resp.encoding or "").lower()
    if not enc or enc == "iso-8859-1":
        try:
            resp.encoding = resp.apparent_encoding or "utf-8"
        except Exception:
            resp.encoding = "utf-8"

    ctype = (resp.headers.get("Content-Type") or "").lower()
    is_html = "text/html" in ctype or "application/xhtml" in ctype or "<html" in resp.text[:1000].lower()

    status = resp.status_code
    html = resp.text if (is_html and isinstance(resp.text, str)) else ""

    if len(html) > MAX_HTML_BYTES:
        html = html[:MAX_HTML_BYTES]
    return final_url, status, html, is_html

def _extract_meta(html: str, name: str):
    m = re.search(
        rf'<meta[^>]+name=["\']{re.escape(name)}["\'][^>]*content=["\'](.*?)["\']',
        html, flags=re.I | re.S
    )
    return m.group(1).strip() if m else None

def _extract_title(html: str):
    m = re.search(r'<title[^>]*>(.*?)</title>', html, flags=re.I | re.S)
    return _clean_text(m.group(1)) if m else None

def _extract_tag(html: str, tag: str):
    m = re.search(rf'<{tag}[^>]*>(.*?)</{tag}>', html, flags=re.I | re.S)
    return _clean_text(m.group(1)) if m else None

def _extract_multi(html: str, tag: str, maxn=8):
    return [
        _clean_text(x)
        for x in re.findall(rf'<{tag}[^>]*>(.*?)</{tag}>', html, flags=re.I | re.S)[:maxn]
    ]

def _extract_paragraphs(html: str, maxn=2, max_chars=400):
    paras = re.findall(r'<p[^>]*>(.*?)</p>', html, flags=re.I | re.S)[:maxn]
    cleaned = []
    for p in paras:
        t = _clean_text(p)
        if not t: continue
        if len(t) > max_chars: t = t[:max_chars]
        cleaned.append(t)
    return cleaned

def _clean_text(s: str):
    if not s: return s
    s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _canonical(html: str):
    m = re.search(
        r'<link[^>]+rel=["\']canonical["\'][^>]*href=["\'](.*?)["\']',
        html, flags=re.I | re.S
    )
    return m.group(1).strip() if m else None

def _robots_meta(html: str):
    m = re.search(
        r'<meta[^>]+name=["\']robots["\'][^>]*content=["\'](.*?)["\']',
        html, flags=re.I | re.S
    )
    if not m: return (False, False)
    content = m.group(1).lower()
    return ("noindex" in content, "nofollow" in content)

def _same_host(a: str, b: str):
    try:
        return _normalize_host(a) == _normalize_host(b)
    except Exception:
        return False

def _extract_links(html: str, base_url: str):
    hrefs = re.findall(r'href\s*=\s*["\']?([^"\' >]+)', html, flags=re.I)
    links = []
    for h in hrefs:
        if not h or h.startswith("#"): continue
        if any(bad in h.lower() for bad in ["javascript:", "(", "=", "{", "}"]): continue
        abs_url = urljoin(base_url, h)
        if abs_url.startswith("http") and not _seems_asset(abs_url):
            links.append(abs_url)
    return links

def crawl_site(start_url: str, max_pages: int = 10, ua: str = None) -> dict:
    if not start_url.startswith(("http://", "https://")):
        start_url = "https://" + start_url

    seen, queue = set(), [start_url]
    pages, quick_wins = [], []
    started = time.time()

    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        if url in seen: continue
        if _seems_asset(url): 
            continue
        seen.add(url)

        try:
            final_url, status, html, is_html = _fetch(url, ua)
        except Exception:
            continue

        title = h1 = meta_desc = canon = None
        h2 = []; h3 = []; paragraphs = []
        noindex = nofollow = False
        issues = []

        if is_html and html:
            title = _extract_title(html)
            h1 = _extract_tag(html, "h1")
            h2 = _extract_multi(html, "h2", maxn=8)
            h3 = _extract_multi(html, "h3", maxn=8)
            meta_desc = _extract_meta(html, "description")
            canon = _canonical(html)
            noindex, nofollow = _robots_meta(html)
            paragraphs = _extract_paragraphs(html, maxn=2, max_chars=400)

            # issues
            if not meta_desc: issues.append("missing_meta_description")
            if not h1: issues.append("missing_h1")
            if canon and _normalize_host(canon.rstrip("/")) != _normalize_host(final_url.rstrip("/")):
                issues.append("canonical_differs")

            # queue nieuwe links (zelfde host, geen assets)
            for link in _extract_links(html, final_url):
                if link not in seen and _same_host(start_url, link) and not _seems_asset(link):
                    queue.append(link)

        pages.append({
            "url": url,
            "final_url": final_url,
            "status": status,
            "title": title,
            "h1": h1,
            "h2": h2,
            "h3": h3,
            "meta_description": meta_desc,
            "canonical": canon,
            "noindex": noindex,
            "nofollow": nofollow,
            "paragraphs": paragraphs,
            "issues": issues
        })

        html = None
        gc.collect()

    dur_ms = int((time.time() - started) * 1000)
    summary = {
        "pages_total": len(pages),
        "ok_200": sum(1 for p in pages if p["status"] == 200),
        "redirect_3xx": sum(1 for p in pages if 300 <= p["status"] < 400),
        "errors_4xx_5xx": sum(1 for p in pages if p["status"] >= 400),
        "fetch_errors": 0,
        "duration_ms": dur_ms,
        "capped_by_runtime": False
    }

    if any("missing_meta_description" in p["issues"] for p in pages):
        quick_wins.append({"type": "missing_meta_description"})
    if any("missing_h1" in p["issues"] for p in pages):
        quick_wins.append({"type": "missing_h1"})
    if any("canonical_differs" in p["issues"] for p in pages):
        quick_wins.append({"type": "canonical_differs"})

    return {
        "start_url": start_url,
        "pages": pages,
        "summary": summary,
        "quick_wins": quick_wins
    }
