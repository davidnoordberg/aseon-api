# crawl_light.py
import re, time, gc
from urllib.parse import urljoin, urlparse
import requests
import os

DEFAULT_TIMEOUT = 10
MAX_HTML_BYTES = int(os.getenv("CRAWL_MAX_HTML_BYTES", "500000"))  # 500 KB cap
HEADERS_TEMPLATE = lambda ua: {"User-Agent": ua or "AseonBot/0.1 (+https://aseon.ai)"}

def _fetch(url: str, ua: str):
    resp = requests.get(url, headers=HEADERS_TEMPLATE(ua), timeout=DEFAULT_TIMEOUT, allow_redirects=True)
    final_url = str(resp.url)
    status = resp.status_code
    html = resp.text if isinstance(resp.text, str) else ""
    if len(html) > MAX_HTML_BYTES:
        html = html[:MAX_HTML_BYTES]
    return final_url, status, html

def _extract_meta(html: str, name: str):
    m = re.search(rf'<meta[^>]+name=["\']{re.escape(name)}["\'][^>]*content=["\'](.*?)["\']', html, flags=re.I|re.S)
    return m.group(1).strip() if m else None

def _extract_title(html: str):
    m = re.search(r'<title[^>]*>(.*?)</title>', html, flags=re.I|re.S)
    return _clean_text(m.group(1)) if m else None

def _extract_tag(html: str, tag: str):
    m = re.search(rf'<{tag}[^>]*>(.*?)</{tag}>', html, flags=re.I|re.S)
    return _clean_text(m.group(1)) if m else None

def _extract_multi(html: str, tag: str, maxn=8):
    return [_clean_text(x) for x in re.findall(rf'<{tag}[^>]*>(.*?)</{tag}>', html, flags=re.I|re.S)[:maxn]]

def _extract_paragraphs(html: str, maxn=2, max_chars=400):
    paras = re.findall(r'<p[^>]*>(.*?)</p>', html, flags=re.I|re.S)[:maxn]
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
    m = re.search(r'<link[^>]+rel=["\']canonical["\'][^>]*href=["\'](.*?)["\']', html, flags=re.I|re.S)
    return m.group(1).strip() if m else None

def _robots_meta(html: str):
    m = re.search(r'<meta[^>]+name=["\']robots["\'][^>]*content=["\'](.*?)["\']', html, flags=re.I|re.S)
    if not m: return (False, False)
    content = m.group(1).lower()
    return ("noindex" in content, "nofollow" in content)

def _same_host(a: str, b: str):
    try:
        return urlparse(a).netloc.split(":")[0].lower() == urlparse(b).netloc.split(":")[0].lower()
    except Exception:
        return False

def crawl_site(start_url: str, max_pages: int = 10, ua: str = None) -> dict:
    if not start_url.startswith(("http://","https://")):
        start_url = "https://" + start_url
    seen, queue = set(), [start_url]
    pages, quick_wins = [], []
    started = time.time()

    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        if url in seen: continue
        seen.add(url)
        try:
            final_url, status, html = _fetch(url, ua)
        except Exception:
            continue

        title = _extract_title(html)
        h1 = _extract_tag(html, "h1")
        h2 = _extract_multi(html, "h2", maxn=8)
        h3 = _extract_multi(html, "h3", maxn=8)
        meta_desc = _extract_meta(html, "description")
        canon = _canonical(html)
        noindex, nofollow = _robots_meta(html)
        paragraphs = _extract_paragraphs(html, maxn=2, max_chars=400)

        # free HTML memory ASAP
        html = None
        gc.collect()

        issues = []
        if not meta_desc: issues.append("missing_meta_description")
        if not h1: issues.append("missing_h1")
        if canon and canon.rstrip("/") != final_url.rstrip("/"):
            issues.append("canonical_differs")

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

        # light discovery
        # NB: we do dit NA het vrijgeven van de grote 'html' string
        # (hier geen html meer gebruiken)

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
        quick_wins.append({"type":"missing_meta_description"})
    if any("missing_h1" in p["issues"] for p in pages):
        quick_wins.append({"type":"missing_h1"})
    if any("canonical_differs" in p["issues"] for p in pages):
        quick_wins.append({"type":"canonical_differs"})

    return {"start_url": start_url, "pages": pages, "summary": summary, "quick_wins": quick_wins}
