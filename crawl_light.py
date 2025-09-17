# crawl_light.py — DOM-order aware crawler met uitgebreide FAQ-extractie
import re, time, gc, json, os
from urllib.parse import urljoin, urlparse
import requests

DEFAULT_TIMEOUT = 10
MAX_HTML_BYTES = int(os.getenv("CRAWL_MAX_HTML_BYTES", "900000"))  # iets ruimer
HEADERS_TEMPLATE = lambda ua: {"User-Agent": ua or "AseonBot/0.4 (+https://aseon.ai)"}

_SKIP_EXT = {
    ".png",".jpg",".jpeg",".webp",".gif",".svg",".ico",".bmp",".avif",
    ".pdf",".zip",".rar",".7z",".gz",".mp4",".mp3",".mov",".wav",".woff",".woff2",".ttf",".eot"
}

def _seems_asset(url: str) -> bool:
    path = urlparse(url).path.lower()
    if any(path.endswith(ext) for ext in _SKIP_EXT):
        return True
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

def _same_host(a: str, b: str):
    try:
        return _normalize_host(a) == _normalize_host(b)
    except Exception:
        return False

def _fetch(url: str, ua: str):
    resp = requests.get(
        url,
        headers=HEADERS_TEMPLATE(ua),
        timeout=DEFAULT_TIMEOUT,
        allow_redirects=True,
    )
    final_url = str(resp.url)

    enc = (resp.encoding or "").lower()
    if not enc or enc == "iso-8859-1":
        try:
            resp.encoding = resp.apparent_encoding or "utf-8"
        except Exception:
            resp.encoding = "utf-8"

    ctype = (resp.headers.get("Content-Type") or "").lower()
    head = (resp.text[:1000].lower() if isinstance(resp.text, str) else "")
    is_html = "text/html" in ctype or "application/xhtml" in ctype or "<html" in head

    status = resp.status_code
    html = resp.text if (is_html and isinstance(resp.text, str)) else ""
    if len(html) > MAX_HTML_BYTES:
        html = html[:MAX_HTML_BYTES]
    return final_url, status, html, is_html

# ---------- helpers ----------
_TAG_RE = re.compile(r"<[^>]+>")
def _clean_text(s: str) -> str:
    if not s: return ""
    s = re.sub(r'<script[^>]*>.*?</script>', ' ', s, flags=re.I|re.S)
    s = re.sub(r'<style[^>]*>.*?</style>', ' ', s, flags=re.I|re.S)
    s = _TAG_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _extract_title(html: str):
    m = re.search(r'<title[^>]*>(.*?)</title>', html, flags=re.I|re.S)
    return _clean_text(m.group(1)) if m else None

def _extract_tag_once(html: str, tag: str):
    m = re.search(rf'<{tag}[^>]*>(.*?)</{tag}>', html, flags=re.I|re.S)
    return _clean_text(m.group(1)) if m else None

def _extract_multi(html: str, tag: str, maxn=50, max_chars=600):
    out = []
    for m in re.finditer(rf'<{tag}[^>]*>(.*?)</{tag}>', html, flags=re.I|re.S):
        t = _clean_text(m.group(1))
        if not t: continue
        if len(t) > max_chars: t = t[:max_chars]
        out.append(t)
        if len(out) >= maxn: break
    return out

def _extract_dom_blocks(html: str, maxn=400, max_chars=700):
    """
    Preserveert DOM-volgorde van relevante elementen voor Q/A-detectie.
    """
    tags = r'(h1|h2|h3|summary|button|dt|dd|li|p)'
    out = []
    for m in re.finditer(rf'<{tags}\b[^>]*>(.*?)</\1>', html, flags=re.I|re.S):
        tag = m.group(1).lower()
        txt = _clean_text(m.group(2))
        if not txt: continue
        if len(txt) > max_chars: txt = txt[:max_chars]
        out.append({"tag": tag, "text": txt})
        if len(out) >= maxn: break
    return out

def _extract_paragraphs(html: str, maxn=200, max_chars=600):
    return _extract_multi(html, "p", maxn=maxn, max_chars=max_chars)

def _extract_jsonld_blocks(html: str):
    blocks = []
    for m in re.finditer(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, flags=re.I|re.S):
        raw = m.group(1).strip()
        try:
            data = json.loads(raw)
            blocks.append(data)
        except Exception:
            continue
    return blocks

def _extract_jsonld_types(data):
    types = set()
    def collect(d):
        if isinstance(d, dict):
            t = d.get("@type")
            if isinstance(t, str): types.add(t)
            elif isinstance(t, list):
                for x in t:
                    if isinstance(x, str): types.add(x)
            for v in d.values(): collect(v)
        elif isinstance(d, list):
            for it in d: collect(it)
    collect(data)
    return sorted(types)

def _canonical(html: str):
    m = re.search(r'<link[^>]+rel=["\']canonical["\'][^>]*href=["\'](.*?)["\']', html, flags=re.I|re.S)
    return m.group(1).strip() if m else None

def _meta_name(html: str, name: str):
    m = re.search(rf'<meta[^>]+name=["\']{re.escape(name)}["\'][^>]*content=["\'](.*?)["\']', html, flags=re.I|re.S)
    return m.group(1).strip() if m else None

def _meta_prop(html: str, prop: str):
    m = re.search(rf'<meta[^>]+property=["\']{re.escape(prop)}["\'][^>]*content=["\'](.*?)["\']', html, flags=re.I|re.S)
    return m.group(1).strip() if m else None

def _robots_meta(html: str):
    m = re.search(r'<meta[^>]+name=["\']robots["\'][^>]*content=["\'](.*?)["\']', html, flags=re.I|re.S)
    if not m: return (False, False)
    c = m.group(1).lower()
    return ("noindex" in c, "nofollow" in c)

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

# ---------- main ----------
def crawl_site(start_url: str, max_pages: int = 12, ua: str = None) -> dict:
    if not start_url.startswith(("http://", "https://")):
        start_url = "https://" + start_url

    seen, queue = set(), [start_url]
    pages, quick_wins = [], []
    started = time.time()
    start_host = start_url

    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        if url in seen: continue
        if _seems_asset(url): continue
        seen.add(url)

        try:
            final_url, status, html, is_html = _fetch(url, ua)
        except Exception:
            continue

        title = h1 = meta_desc = canon = None
        noindex = nofollow = False
        issues = []
        og_title = og_desc = twitter_card = None
        jsonld_types = []
        dom_blocks = []
        h2 = []; h3 = []; paragraphs = []; lis = []; summaries = []; buttons = []; dts = []; dds = []
        outlinks = []
        faq_jsonld = None
        word_count = 0

        if is_html and html:
            title = _extract_title(html)
            h1 = _extract_tag_once(html, "h1")
            h2 = _extract_multi(html, "h2", maxn=50)
            h3 = _extract_multi(html, "h3", maxn=50)
            paragraphs = _extract_paragraphs(html, maxn=250)
            lis = _extract_multi(html, "li", maxn=250)
            summaries = _extract_multi(html, "summary", maxn=250)
            buttons = _extract_multi(html, "button", maxn=250)
            dts = _extract_multi(html, "dt", maxn=250)
            dds = _extract_multi(html, "dd", maxn=250)

            dom_blocks_tagged = _extract_dom_blocks(html, maxn=600)
            dom_blocks = [x["text"] for x in dom_blocks_tagged]

            meta_desc = _meta_name(html, "description")
            og_title = _meta_prop(html, "og:title")
            og_desc  = _meta_prop(html, "og:description")
            twitter_card = _meta_prop(html, "twitter:card") or _meta_name(html, "twitter:card")
            canon = _canonical(html)
            noindex, nofollow = _robots_meta(html)

            # JSON-LD parsing
            ld_blocks = _extract_jsonld_blocks(html)
            all_types = set()
            for block in ld_blocks:
                for t in _extract_jsonld_types(block):
                    all_types.add(t)
            jsonld_types = sorted(all_types)

            # only FAQ-relevante blokken meesturen
            faq_candidates = []
            def looks_like_faq(obj):
                if isinstance(obj, dict):
                    t = obj.get("@type")
                    if (isinstance(t, str) and t.lower() == "faqpage") or ("mainEntity" in obj):
                        return True
                return False
            for b in ld_blocks:
                if looks_like_faq(b):
                    faq_candidates.append(b)
                elif isinstance(b, dict) and "@graph" in b and isinstance(b["@graph"], list):
                    for node in b["@graph"]:
                        if looks_like_faq(node):
                            faq_candidates.append(node)
            if faq_candidates:
                try:
                    faq_jsonld = json.dumps(faq_candidates)  # als string opslaan
                except Exception:
                    faq_jsonld = None

            # links
            outlinks = [l for l in _extract_links(html, final_url) if _same_host(start_host, l)]

            # zichtbare woorden
            word_count = len(_clean_text(html).split())

            # issues
            if not meta_desc: issues.append("missing_meta_description")
            if not h1: issues.append("missing_h1")
            if canon:
                try:
                    from urllib.parse import urlsplit
                    if urlsplit(canon).netloc.lower().lstrip("www.") != urlsplit(final_url).netloc.lower().lstrip("www."):
                        issues.append("canonical_differs")
                except Exception:
                    pass
            if not title: issues.append("missing_title")

            for link in outlinks:
                if link not in seen and not _seems_asset(link):
                    queue.append(link)

        pages.append({
            "url": url,
            "final_url": final_url,
            "status": status,
            "title": title,
            "h1": h1,
            "h2": h2,
            "h3": h3,
            "paragraphs": paragraphs,      # nu ALLE p's (tot 250)
            "li": lis,
            "summary": summaries,
            "buttons": buttons,
            "dt": dts,
            "dd": dds,
            "dom_blocks": dom_blocks,      # cruciaal: DOM-volgorde
            "meta_description": meta_desc,
            "og_title": og_title,
            "og_description": og_desc,
            "twitter_card": twitter_card,
            "canonical": canon,
            "noindex": noindex,
            "nofollow": nofollow,
            "jsonld_types": jsonld_types,
            "faq_jsonld": faq_jsonld,      # stringified JSON
            "links": outlinks,
            "word_count": word_count,
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
    }

    quick_wins = []
    if any("missing_meta_description" in p["issues"] for p in pages):
        quick_wins.append({"type":"missing_meta_description"})
    if any("missing_h1" in p["issues"] for p in pages):
        quick_wins.append({"type":"missing_h1"})
    if any("canonical_differs" in p["issues"] for p in pages):
        quick_wins.append({"type":"canonical_differs"})

    return {
        "start_url": start_url,
        "pages": pages,
        "summary": summary,
        "quick_wins": quick_wins
    }
