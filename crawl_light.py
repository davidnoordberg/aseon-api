# crawl_light.py
import os
import re
import gc
import time
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode
import requests

DEFAULT_TIMEOUT = float(os.getenv("CRAWL_TIMEOUT_SEC", "10"))
MAX_HTML_BYTES  = int(os.getenv("CRAWL_MAX_HTML_BYTES", "600000"))  # 600 KB cap
MAX_LINKS_PAGE  = int(os.getenv("CRAWL_MAX_LINKS_PER_PAGE", "150"))
USER_AGENT      = os.getenv("CRAWL_USER_AGENT", "AseonBot/0.2 (+https://aseon.ai)")

HEADERS = {"User-Agent": USER_AGENT}

JS_LIKE = re.compile(r"^(javascript:|void\(|#|mailto:|tel:|data:)", re.I)
BAD_CHUNKS = ("a.getAttribute(", "window.", "document.", "onclick=", "return false")
REL_CANON = re.compile(r'<link[^>]+rel=["\']?canonical["\']?[^>]+href=["\']?([^"\'>\s]+)', re.I)
META_DESC = re.compile(r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']', re.I|re.S)
TITLE_RX  = re.compile(r'<title[^>]*>(.*?)</title>', re.I|re.S)
H1_RX     = re.compile(r'<h1[^>]*>(.*?)</h1>', re.I|re.S)
H2_RX     = re.compile(r'<h2[^>]*>(.*?)</h2>', re.I|re.S)
H3_RX     = re.compile(r'<h3[^>]*>(.*?)</h3>', re.I|re.S)
P_RX      = re.compile(r'<p[^>]*>(.*?)</p>', re.I|re.S)
ROBOTS_RX = re.compile(r'<meta[^>]+name=["\']robots["\'][^>]+content=["\'](.*?)["\']', re.I|re.S)
HREF_RX   = re.compile(r'href\s*=\s*["\']?([^"\' >]+)', re.I)

UTM_KEYS = {"utm_source","utm_medium","utm_campaign","utm_term","utm_content","utm_id","gclid","fbclid","mc_cid","mc_eid"}

def _clean_text(s: str) -> str:
    if not s: return ""
    s = re.sub(r'<script[\s\S]*?</script>', ' ', s, flags=re.I)
    s = re.sub(r'<style[\s\S]*?</style>', ' ', s, flags=re.I)
    s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def _fetch(url: str):
    r = requests.get(url, headers=HEADERS, timeout=DEFAULT_TIMEOUT, allow_redirects=True)
    html = r.text if isinstance(r.text, str) else ""
    if len(html) > MAX_HTML_BYTES:
        html = html[:MAX_HTML_BYTES]
    return str(r.url), r.status_code, html

def _robots_meta(html: str):
    m = ROBOTS_RX.search(html or "")
    if not m: return (False, False)
    c = m.group(1).lower()
    return ("noindex" in c, "nofollow" in c)

def _canonical(html: str):
    m = REL_CANON.search(html or "")
    return m.group(1).strip() if m else None

def _extract_multi(rx, html, maxn=8):
    return [_clean_text(x) for x in rx.findall(html or "")[:maxn]]

def _extract_paragraphs(html: str, maxn=2, max_chars=400):
    out = []
    for p in P_RX.findall(html or "")[:maxn]:
        t = _clean_text(p)
        if not t: continue
        if len(t) > max_chars: t = t[:max_chars]
        out.append(t)
    return out

def _same_host(a: str, b: str) -> bool:
    try:
        ha = urlparse(a).netloc.split(":")[0].lower().lstrip("www.")
        hb = urlparse(b).netloc.split(":")[0].lower().lstrip("www.")
        return ha == hb
    except Exception:
        return False

def _normalize(u: str) -> str:
    """normalize URL (strip utm, sort query, drop fragments, unify www)"""
    try:
        p = urlparse(u)
        q = [(k, v) for (k,v) in parse_qsl(p.query, keep_blank_values=True) if k not in UTM_KEYS]
        host = p.netloc.lower()
        host = host.replace("www.", "")  # normalize
        new = p._replace(netloc=host, fragment="", query=urlencode(sorted(q)))
        return urlunparse(new).rstrip("/")
    except Exception:
        return u.rstrip("/")

def _extract_links(html: str, base_url: str):
    links = []
    for raw in HREF_RX.findall(html or ""):
        raw = raw.strip()
        if not raw or JS_LIKE.match(raw): 
            continue
        if any(b in raw for b in BAD_CHUNKS):
            continue
        absu = urljoin(base_url, raw)
        if absu.startswith("http"):
            links.append(absu)
        if len(links) >= MAX_LINKS_PAGE:
            break
    return links

def crawl_site(start_url: str, max_pages: int = 10, ua: str = None) -> dict:
    if not start_url.startswith(("http://","https://")):
        start_url = "https://" + start_url
    start_url = _normalize(start_url)

    seen, q = set(), [start_url]
    pages, quick_wins = [], []
    started = time.time()

    while q and len(pages) < max_pages:
        url = q.pop(0)
        if url in seen: 
            continue
        seen.add(url)

        try:
            final_url, status, html = _fetch(url)
        except Exception:
            continue

        canon = _canonical(html)
        if canon:
            canon_n = _normalize(urljoin(final_url, canon))
            final_n = _normalize(final_url)
            if canon_n != final_n:
                # treat canonical as authoritative target
                final_url = canon_n

        title = _clean_text(TITLE_RX.search(html or "") .group(1)) if TITLE_RX.search(html or "") else None
        h1    = _clean_text(H1_RX.search(html or "")    .group(1)) if H1_RX.search(html or "")    else None
        h2    = _extract_multi(H2_RX, html, 12)
        h3    = _extract_multi(H3_RX, html, 12)
        meta_desc = META_DESC.search(html or "")
        meta_desc = _clean_text(meta_desc.group(1)) if meta_desc else None
        noindex, nofollow = _robots_meta(html)
        paragraphs = _extract_paragraphs(html, 3, 500)

        issues = []
        if not meta_desc: issues.append("missing_meta_description")
        if not h1: issues.append("missing_h1")
        if canon and _normalize(urljoin(final_url, canon)) != _normalize(final_url):
            issues.append("canonical_differs")
        if status >= 400: issues.append(f"http_{status}")

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
            "issues": issues,
        })

        # link discovery (respect nofollow = don't add outlinks from this page)
        if not nofollow and status == 200:
            for link in _extract_links(html, final_url):
                link_n = _normalize(link)
                if link_n not in seen and _same_host(start_url, link_n):
                    q.append(link_n)

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
        "duration_ms": dur_ms,
        "capped_by_runtime": len(pages) >= max_pages,
    }

    if any("missing_meta_description" in p["issues"] for p in pages):
        quick_wins.append({"type":"missing_meta_description"})
    if any("missing_h1" in p["issues"] for p in pages):
        quick_wins.append({"type":"missing_h1"})
    if any("canonical_differs" in p["issues"] for p in pages):
        quick_wins.append({"type":"canonical_differs"})

    return {"start_url": start_url, "pages": pages, "summary": summary, "quick_wins": quick_wins}
