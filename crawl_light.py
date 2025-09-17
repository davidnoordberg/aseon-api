# crawl_light.py — HTML-only crawler met FAQ/accordion-detectie en faq_visible output
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

_FAQ_CLASS_RE = re.compile(r'class\s*=\s*["\'][^"\']*(faq|accordion|accordeon|question|qna|qa|toggle|collapsible|expander)[^"\']*["\']', re.I)
_TAG_RE = re.compile(r"<[^>]+>")

def _seems_asset(url: str) -> bool:
    path = urlparse(url).path.lower()
    if any(path.endswith(ext) for ext in _SKIP_EXT): return True
    if "favicon" in path or "apple-touch-icon" in path: return True
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
    resp = requests.get(url, headers=HEADERS_TEMPLATE(ua), timeout=DEFAULT_TIMEOUT, allow_redirects=True)
    final_url = str(resp.url)

    enc = (resp.encoding or "").lower()
    if not enc or enc == "iso-8859-1":
        try:
            resp.encoding = resp.apparent_encoding or "utf-8"
        except Exception:
            resp.encoding = "utf-8"

    ctype = (resp.headers.get("Content-Type") or "").lower()
    text_head = (resp.text[:1000].lower() if isinstance(resp.text, str) else "")
    is_html = "text/html" in ctype or "application/xhtml" in ctype or "<html" in text_head

    status = resp.status_code
    html = resp.text if (is_html and isinstance(resp.text, str)) else ""
    if len(html) > MAX_HTML_BYTES:
        html = html[:MAX_HTML_BYTES]
    return final_url, status, html, is_html

def _clean_text(s: str):
    if not s: return s
    s = re.sub(r'<script[^>]*>.*?</script>', ' ', s, flags=re.I|re.S)
    s = re.sub(r'<style[^>]*>.*?</style>', ' ', s, flags=re.I|re.S)
    s = _TAG_RE.sub(' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _extract_meta(html: str, name: str):
    m = re.search(rf'<meta[^>]+name=["\']{re.escape(name)}["\'][^>]*content=["\'](.*?)["\']', html, flags=re.I|re.S)
    return (m.group(1).strip() if m else None)

def _extract_meta_property(html: str, prop: str):
    m = re.search(rf'<meta[^>]+property=["\']{re.escape(prop)}["\'][^>]*content=["\'](.*?)["\']', html, flags=re.I|re.S)
    return (m.group(1).strip() if m else None)

def _extract_title(html: str):
    m = re.search(r'<title[^>]*>(.*?)</title>', html, flags=re.I|re.S)
    return _clean_text(m.group(1)) if m else None

def _extract_tag(html: str, tag: str):
    m = re.search(rf'<{tag}[^>]*>(.*?)</{tag}>', html, flags=re.I|re.S)
    return _clean_text(m.group(1)) if m else None

def _extract_multi(html: str, tag: str, maxn=20):
    return [
        _clean_text(x)
        for x in re.findall(rf'<{tag}[^>]*>(.*?)</{tag}>', html, flags=re.I|re.S)[:maxn]
    ]

def _extract_paragraphs(html: str, maxn=40, max_chars=800):
    paras = re.findall(r'<p[^>]*>(.*?)</p>', html, flags=re.I|re.S)[:maxn]
    cleaned = []
    for p in paras:
        t = _clean_text(p)
        if not t: continue
        if len(t) > max_chars: t = t[:max_chars]
        cleaned.append(t)
    return cleaned

def _extract_list_items(html: str, maxn=80, max_chars=600):
    items = re.findall(r'<li[^>]*>(.*?)</li>', html, flags=re.I|re.S)[:maxn]
    out = []
    for it in items:
        t = _clean_text(it)
        if not t: continue
        if len(t) > max_chars: t = t[:max_chars]
        out.append(t)
    return out

def _extract_buttons(html: str, maxn=80):
    btns = re.findall(r'<button[^>]*>(.*?)</button>', html, flags=re.I|re.S)[:maxn]
    btns += re.findall(r'<a[^>]+role=["\']button["\'][^>]*>(.*?)</a>', html, flags=re.I|re.S)[:maxn]
    return [_clean_text(b) for b in btns if _clean_text(b)]

def _extract_summaries(html: str, maxn=80):
    sums = re.findall(r'<summary[^>]*>(.*?)</summary>', html, flags=re.I|re.S)[:maxn]
    return [_clean_text(s) for s in sums if _clean_text(s)]

def _extract_dt_dd_pairs(html: str, maxn=200):
    pairs = re.findall(r'<dt[^>]*>(.*?)</dt>\s*<dd[^>]*>(.*?)</dd>', html, flags=re.I|re.S)[:maxn]
    out = []
    for q, a in pairs:
        q2, a2 = _clean_text(q), _clean_text(a)
        if q2 and a2: out.append({"q": q2, "a": a2})
    return out

def _extract_details_pairs(html: str, maxn=200):
    blocks = re.findall(r'<details[^>]*>(.*?)</details>', html, flags=re.I|re.S)[:maxn]
    out = []
    for b in blocks:
        sm = re.search(r'<summary[^>]*>(.*?)</summary>(.*)$', b, flags=re.I|re.S)
        if not sm: continue
        q = _clean_text(sm.group(1))
        a = _clean_text(sm.group(2))
        if q and a: out.append({"q": q, "a": a})
    return out

def _extract_accordion_pairs(html: str, maxn=200):
    out = []
    # Pak containers met typische FAQ/accordion class
    for m in re.finditer(r'<(div|li|section)[^>]*>(.*?)</\1>', html, flags=re.I|re.S):
        block = m.group(0)
        if not _FAQ_CLASS_RE.search(block): 
            continue
        # vraag-kandidaten in block
        q_candidates = []
        for tag in ("summary","h1","h2","h3","h4","button","strong","span"):
            for qx in re.findall(rf'<{tag}[^>]*>(.*?)</{tag}>', block, flags=re.I|re.S):
                t = _clean_text(qx)
                if t and (t.endswith("?") or len(t.split()) <= 14):  # korte heading of eindigt op ?
                    q_candidates.append(t)
        q = q_candidates[0] if q_candidates else None
        if not q: 
            continue
        # antwoord = block zonder de eerste vraag-node
        after = re.sub(r'<summary[^>]*>.*?</summary>', ' ', block, flags=re.I|re.S, count=1)
        for tag in ("h1","h2","h3","h4","button","strong","span"):
            after = re.sub(rf'<{tag}[^>]*>.*?</{tag}>', ' ', after, flags=re.I|re.S, count=1)
        a = _clean_text(after)
        if a and len(a.split()) >= 6:
            out.append({"q": q, "a": a})
    return out

def _extract_heading_follow_pairs(html: str, maxn=120):
    out = []
    # Heuristic: <h3>Vraag?</h3><p>Antwoord...</p>
    for tag in ("h2","h3","h4"):
        for m in re.finditer(rf'<{tag}[^>]*>(.*?)</{tag}>\s*((?:<(?!/?h[1-6]).*?>.*?)+)', html, flags=re.I|re.S):
            q = _clean_text(m.group(1))
            following = _clean_text(m.group(2))
            if q and (q.endswith("?") or len(q.split()) <= 14):
                # pak eerste ~120 woorden als antwoord
                words = following.split()
                if len(words) >= 6:
                    a = " ".join(words[:2000])  # ruim; AEO knipt later
                    out.append({"q": q, "a": a})
            if len(out) >= maxn: break
    return out

def _visible_text_word_count(html: str) -> int:
    t = _clean_text(html or "")
    if not t: return 0
    return len([w for w in t.split(" ") if w])

def _canonical(html: str):
    m = re.search(r'<link[^>]+rel=["\']canonical["\'][^>]*href=["\'](.*?)["\']', html, flags=re.I|re.S)
    return m.group(1).strip() if m else None

def _robots_meta(html: str):
    m = re.search(r'<meta[^>]+name=["\']robots["\'][^>]*content=["\'](.*?)["\']', html, flags=re.I|re.S)
    if not m: return (False, False)
    content = m.group(1).lower()
    return ("noindex" in content, "nofollow" in content)

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

def _extract_jsonld_blocks(html: str):
    blocks = []
    for m in re.finditer(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, flags=re.I|re.S):
        raw = m.group(1).strip()
        try:
            data = json.loads(raw)
            blocks.append(data)
        except Exception:
            # probeer JSON die arrays concateneert te repareren niet — sla over
            continue
    return blocks

def _jsonld_types(data):
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
    return sorted(types)[:20]

def _extract_faq_jsonld(html: str):
    out = []
    for data in _extract_jsonld_blocks(html):
        # neem alleen blokken die FAQPage bevatten
        types = set([t.lower() for t in _jsonld_types(data)])
        if "faqpage" in types:
            out.append(data)
    if not out:
        return None
    return out if len(out) > 1 else out[0]

def _dedupe_qas(qas):
    seen_q = set()
    out = []
    for qa in qas:
        q = (qa.get("q") or "").strip()
        a = (qa.get("a") or "").strip()
        if not q or not a: continue
        key = (q.lower(), a[:160].lower())
        if key in seen_q: continue
        seen_q.add(key)
        out.append({"q": q, "a": a})
    return out

def _extract_faq_pairs(html: str):
    qas = []
    qas += _extract_details_pairs(html)
    qas += _extract_dt_dd_pairs(html)
    qas += _extract_accordion_pairs(html)
    qas += _extract_heading_follow_pairs(html)
    # Laatste redmiddel: summary + eerstvolgende paragraaf
    sums = re.findall(r'<summary[^>]*>(.*?)</summary>\s*((?:<p[^>]*>.*?</p>)+)', html, flags=re.I|re.S)
    for q, a in sums:
        q2, a2 = _clean_text(q), _clean_text(a)
        if q2 and a2 and len(a2.split()) >= 6:
            qas.append({"q": q2, "a": a2})
    return _dedupe_qas(qas)

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
        h2 = []; h3 = []; h4 = []
        paragraphs = []; lis = []; summaries = []; buttons = []
        dt_list = []; dd_list = []
        noindex = nofollow = False
        issues = []
        og_title = og_desc = twitter_card = None
        jsonld_types = []
        outlinks = []
        word_count = 0
        faq_visible = []
        faq_jsonld = None

        if is_html and html:
            title = _extract_title(html)
            h1 = _extract_tag(html, "h1")
            h2 = _extract_multi(html, "h2", maxn=20)
            h3 = _extract_multi(html, "h3", maxn=20)
            h4 = _extract_multi(html, "h4", maxn=20)
            meta_desc = _extract_meta(html, "description")
            og_title = _extract_meta_property(html, "og:title")
            og_desc  = _extract_meta_property(html, "og:description")
            twitter_card = _extract_meta_property(html, "twitter:card") or _extract_meta(html, "twitter:card")
            canon = _canonical(html)
            noindex, nofollow = _robots_meta(html)
            paragraphs = _extract_paragraphs(html, maxn=40, max_chars=1000)
            lis = _extract_list_items(html, maxn=120, max_chars=800)
            summaries = _extract_summaries(html, maxn=120)
            buttons = _extract_buttons(html, maxn=120)

            # definities afzonderlijk (handig voor zichtbare blokken)
            dt_list = [ _clean_text(x) for x in re.findall(r'<dt[^>]*>(.*?)</dt>', html, flags=re.I|re.S) ]
            dd_list = [ _clean_text(x) for x in re.findall(r'<dd[^>]*>(.*?)</dd>', html, flags=re.I|re.S) ]

            # FAQ’s uit zichtbare HTML-structuren
            faq_visible = _extract_faq_pairs(html)

            # JSON-LD types + FAQ JSON-LD object
            jl_blocks = _extract_jsonld_blocks(html)
            all_types = set()
            for b in jl_blocks:
                for t in _jsonld_types(b):
                    all_types.add(t)
            jsonld_types = sorted(all_types)[:20]
            faq_jsonld = _extract_faq_jsonld(html)

            outlinks = [l for l in _extract_links(html, final_url) if _same_host(start_host, l)]
            word_count = _visible_text_word_count(html)

            # eenvoudige issues
            if not meta_desc: issues.append("missing_meta_description")
            if not h1: issues.append("missing_h1")
            if canon and _normalize_host(canon.rstrip("/")) != _normalize_host(final_url.rstrip("/")):
                issues.append("canonical_differs")
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
            "h4": h4,
            "meta_description": meta_desc,
            "og_title": og_title,
            "og_description": og_desc,
            "twitter_card": twitter_card,
            "canonical": canon,
            "noindex": noindex,
            "nofollow": nofollow,
            "paragraphs": paragraphs,
            "li": lis,
            "summary": summaries,
            "buttons": buttons,
            "dt": dt_list,
            "dd": dd_list,
            "faq_visible": faq_visible,     # <— SUPPLY aan AEO-agent
            "faq_jsonld": faq_jsonld,       # <— SUPPLY aan AEO-agent
            "jsonld_types": jsonld_types,
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
        "capped_by_runtime": False
    }

    quick_wins = []
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
