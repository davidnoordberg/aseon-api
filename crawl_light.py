# crawl_light.py
import os, re, time, gc
from urllib.parse import urljoin, urlparse, urldefrag
import requests

DEFAULT_TIMEOUT = 10
MAX_HTML_BYTES = int(os.getenv("CRAWL_MAX_HTML_BYTES", "600000"))  # 600 KB
HEADERS_TEMPLATE = lambda ua: {"User-Agent": ua or "AseonBot/0.2 (+https://aseon.ai)"}
ASSET_EXT = (".png",".jpg",".jpeg",".webp",".gif",".svg",".ico",".css",".js",".mp4",".webm",".pdf",".woff",".woff2",".ttf")

def _fetch(url: str, ua: str):
    resp = requests.get(url, headers=HEADERS_TEMPLATE(ua), timeout=DEFAULT_TIMEOUT, allow_redirects=True)
    final_url = str(resp.url)
    status = resp.status_code
    # meet TTFB (approx: requests' elapsed)
    ttfb_ms = int(resp.elapsed.total_seconds() * 1000)
    # alleen tekst/html meenemen
    ctype = resp.headers.get("content-type","").lower()
    html = resp.text if ("text/html" in ctype or "application/xhtml" in ctype) else ""
    if len(html) > MAX_HTML_BYTES: html = html[:MAX_HTML_BYTES]
    return final_url, status, html, ttfb_ms, int(resp.headers.get("content-length") or len(html) or 0)

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
    out = []
    for p in paras:
        t = _clean_text(p)
        if not t: continue
        if len(t) > max_chars: t = t[:max_chars]
        out.append(t)
    return out

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

def _looks_asset(url: str):
    path = urlparse(url).path.lower()
    return path.endswith(ASSET_EXT)

def _extract_links(html: str, base_url: str, host_origin: str, cap: int):
    # href=…, met of zonder quotes; minimalistisch en snel
    raw = re.findall(r'href\s*=\s*["\']?([^"\' >]+)', html, flags=re.I)
    links = []
    for h in raw:
        # strip fragmenten (#…)
        h, _ = urldefrag(h)
        if not h or h.startswith(("mailto:","tel:","javascript:")): continue
        absu = urljoin(base_url, h)
        if not absu.startswith("http"): continue
        if _looks_asset(absu): continue
        # blijf op host
        if not _same_host(host_origin, absu): continue
        links.append(absu)
        if len(links) >= cap: break
    return links

def crawl_site(start_url: str, max_pages: int = 10, ua: str = None) -> dict:
    if not start_url.startswith(("http://","https://")):
        start_url = "https://" + start_url
    host_origin = start_url
    seen, queue = set(), [start_url]
    pages, quick_wins = [], []
    started = time.time()

    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        if url in seen: continue
        seen.add(url)

        try:
            final_url, status, html, ttfb_ms, content_bytes = _fetch(url, ua)
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

        issues = []
        if not meta_desc: issues.append("missing_meta_description")
        if not h1: issues.append("missing_h1")
        if canon and canon.rstrip("/") != final_url.rstrip("/"):
            issues.append("canonical_differs")

        # linkdiscovery
        internal_links, external_links = [], []
        if html and not nofollow and status == 200:
            discovered = _extract_links(html, final_url, host_origin, cap=max(100, max_pages*10))
            for l in discovered:
                (internal_links if _same_host(final_url, l) else external_links).append(l)
                if l not in seen and l not in queue and _same_host(final_url, l):
                    if len(queue) < max_pages * 5:
                        queue.append(l)

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
            "internal_links": list(dict.fromkeys(internal_links))[:50],
            "external_links": list(dict.fromkeys(external_links))[:50],
            "ttfb_ms": ttfb_ms,
            "content_bytes": content_bytes,
            "issues": issues
        })

        # free memory
        html = None
        gc.collect()

    dur_ms = int((time.time() - started) * 1000)
    summary = {
        "pages_total": len(pages),
        "ok_200": sum(1 for p in pages if p["status"] == 200),
        "redirect_3xx": sum(1 for p in pages if 300 <= p["status"] < 400),
        "errors_4xx_5xx": sum(1 for p in pages if p["status"] >= 400),
        "fetch_errors": 0,
        "avg_ttfb_ms": int(sum(p["ttfb_ms"] for p in pages)/len(pages)) if pages else 0,
        "avg_bytes": int(sum(p["content_bytes"] for p in pages)/len(pages)) if pages else 0,
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
