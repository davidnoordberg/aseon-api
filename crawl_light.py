#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re, time, gc, json, os, sys, argparse
from urllib.parse import urljoin, urlparse
import requests

DEFAULT_TIMEOUT = 12
MAX_HTML_BYTES = int(os.getenv("CRAWL_MAX_HTML_BYTES", "700000"))  # 700 KB
DEFAULT_UA = os.getenv("CRAWL_UA", "AseonBot/0.4 (+https://aseon.ai)")
HEADERS_TEMPLATE = lambda ua: {"User-Agent": ua or DEFAULT_UA, "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8"}

_SKIP_EXT = {
    ".png",".jpg",".jpeg",".webp",".gif",".svg",".ico",".bmp",".avif",
    ".pdf",".zip",".rar",".7z",".gz",".mp4",".mp3",".mov",".wav",".woff",".woff2",".ttf",".eot",".otf",".css",".js"
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
    text_head = (resp.text[:1000].lower() if isinstance(resp.text, str) else "")
    is_html = "text/html" in ctype or "application/xhtml" in ctype or "<html" in text_head

    status = resp.status_code
    html = resp.text if (is_html and isinstance(resp.text, str)) else ""
    if isinstance(html, str) and len(html) > MAX_HTML_BYTES:
        html = html[:MAX_HTML_BYTES]
    return final_url, status, html, is_html, ctype

def _clean_text(s: str):
    if not s: return s
    s = re.sub(r'<script[^>]*>.*?</script>', ' ', s, flags=re.I|re.S)
    s = re.sub(r'<style[^>]*>.*?</style>', ' ', s, flags=re.I|re.S)
    s = re.sub(r'<!--.*?-->', ' ', s, flags=re.S)
    s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _extract_meta(html: str, name: str):
    m = re.search(
        rf'<meta[^>]+name=["\']{re.escape(name)}["\'][^>]*content=["\'](.*?)["\']',
        html, flags=re.I | re.S
    )
    return (m.group(1).strip() if m else None)

def _extract_meta_property(html: str, prop: str):
    m = re.search(
        rf'<meta[^>]+property=["\']{re.escape(prop)}["\'][^>]*content=["\'](.*?)["\']',
        html, flags=re.I | re.S
    )
    return (m.group(1).strip() if m else None)

def _extract_title(html: str):
    m = re.search(r'<title[^>]*>(.*?)</title>', html, flags=re.I | re.S)
    return _clean_text(m.group(1)) if m else None

def _extract_tag(html: str, tag: str):
    m = re.search(rf'<{tag}[^>]*>(.*?)</{tag}>', html, flags=re.I | re.S)
    return _clean_text(m.group(1)) if m else None

def _extract_multi(html: str, tag: str, maxn=8):
    out = []
    for x in re.findall(rf'<{tag}[^>]*>(.*?)</{tag}>', html, flags=re.I | re.S)[:maxn]:
        t = _clean_text(x)
        if t: out.append(t)
    return out

def _extract_paragraphs(html: str, maxn=3, max_chars=500):
    paras = re.findall(r'<p[^>]*>(.*?)</p>', html, flags=re.I | re.S)[:maxn]
    cleaned = []
    for p in paras:
        t = _clean_text(p)
        if not t: continue
        if len(t) > max_chars: t = t[:max_chars]
        cleaned.append(t)
    return cleaned

def _visible_text_word_count(html: str) -> int:
    t = _clean_text(html or "")
    if not t: return 0
    return len([w for w in t.split(" ") if w])

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

def _extract_jsonld_types(html: str):
    types = set()
    for m in re.finditer(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, flags=re.I|re.S):
        block = m.group(1).strip()
        try:
            data = json.loads(block)
        except Exception:
            continue
        def collect(d):
            if isinstance(d, dict):
                t = d.get("@type")
                if isinstance(t, str):
                    types.add(t)
                elif isinstance(t, list):
                    for x in t:
                        if isinstance(x, str): types.add(x)
                for v in d.values(): collect(v)
            elif isinstance(d, list):
                for it in d: collect(it)
        collect(data)
    return sorted(types)[:20]

def _extract_jsonld_blocks(html: str):
    blocks = []
    for m in re.finditer(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, flags=re.I|re.S):
        raw = (m.group(1) or "").strip()
        if not raw: continue
        try:
            data = json.loads(raw)
        except Exception:
            # soms meerdere JSON's concatenated; brute try split
            try:
                # naive: probeer array met meerdere objects te parsen
                raw2 = raw.strip()
                if raw2.startswith("{") and raw2.endswith("}"):
                    data = json.loads(raw2)
                elif raw2.startswith("[") and raw2.endswith("]"):
                    data = json.loads(raw2)
                else:
                    continue
            except Exception:
                continue
        blocks.append(data)
    return blocks

def _collect_faq_from_jsonld(data):
    qa = []
    def collect(d):
        if isinstance(d, dict):
            t = d.get("@type")
            types = [t] if isinstance(t, str) else (t or [])
            if "FAQPage" in types:
                ents = d.get("mainEntity") or []
                if isinstance(ents, dict): ents = [ents]
                for ent in ents:
                    if not isinstance(ent, dict): continue
                    q = ent.get("name") or ent.get("headline") or ""
                    ans = ent.get("acceptedAnswer") or {}
                    if isinstance(ans, list): ans = (ans[0] if ans else {})
                    a = ""
                    if isinstance(ans, dict):
                        a = ans.get("text") or ans.get("articleBody") or ""
                    q = re.sub(r'\s+', ' ', _clean_text(q or ""))
                    a = re.sub(r'\s+', ' ', _clean_text(a or ""))
                    if q and a:
                        qa.append({"q": q, "a": a})
            for v in d.values(): collect(v)
        elif isinstance(d, list):
            for it in d: collect(it)
    collect(data)
    return qa

def _extract_visible_faq(html: str, maxn=30):
    qa = []
    # details/summary
    for m in re.finditer(r'<details[^>]*>\s*<summary[^>]*>(.*?)</summary>(.*?)</details>', html, flags=re.I|re.S):
        q = re.sub(r'\s+', ' ', _clean_text(m.group(1)))
        a = re.sub(r'\s+', ' ', _clean_text(m.group(2)))
        if q and a: qa.append({"q": q, "a": a})
        if len(qa) >= maxn: return qa[:maxn]
    # dl/dt/dd
    for m in re.finditer(r'<dt[^>]*>(.*?)</dt>\s*<dd[^>]*>(.*?)</dd>', html, flags=re.I|re.S):
        q = re.sub(r'\s+', ' ', _clean_text(m.group(1)))
        a = re.sub(r'\s+', ' ', _clean_text(m.group(2)))
        if q and a: qa.append({"q": q, "a": a})
        if len(qa) >= maxn: return qa[:maxn]
    # headings + volgend blok
    for hx in ["h2","h3","h4"]:
        pattern = rf'<{hx}[^>]*>(.*?)</{hx}>\s*(<(?:p|div|section|article|ul|ol)[^>]*>.*?</(?:p|div|section|article|ul|ol)>)'
        for m in re.finditer(pattern, html, flags=re.I|re.S):
            q = re.sub(r'\s+', ' ', _clean_text(m.group(1)))
            a = re.sub(r'\s+', ' ', _clean_text(m.group(2)))
            if q and a and len(q) > 8 and len(a) > 20:
                qa.append({"q": q, "a": a})
                if len(qa) >= maxn: return qa[:maxn]
    # simpele Q: / A: patroon
    for m in re.finditer(r'(?:<p[^>]*>|<li[^>]*>)(\s*Q[:\-\s]+.*?)(?:</p>|</li>).*?(?:<p[^>]*>|<li[^>]*>)(\s*A[:\-\s]+.*?)(?:</p>|</li>)', html, flags=re.I|re.S):
        q = re.sub(r'^\s*Q[:\-\s]+', '', _clean_text(m.group(1)))
        a = re.sub(r'^\s*A[:\-\s]+', '', _clean_text(m.group(2)))
        if q and a: qa.append({"q": q, "a": a})
        if len(qa) >= maxn: return qa[:maxn]
    return qa[:maxn]

def _try_js_render(url: str, ua: str, timeout_ms=15000):
    # Alleen gebruiken als Playwright beschikbaar is
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return ""
    html = ""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(user_agent=ua or DEFAULT_UA)
            page = ctx.new_page()
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            # probeer accordion content te tonen
            page.evaluate("""
              document.querySelectorAll('details').forEach(d => d.open = true);
              document.querySelectorAll('[aria-controls], [data-accordion-target]').forEach(el => { try { el.click(); } catch(e){} });
            """)
            html = page.content()
            browser.close()
    except Exception:
        return ""
    if isinstance(html, str) and len(html) > MAX_HTML_BYTES:
        html = html[:MAX_HTML_BYTES]
    return html

def crawl_site(start_url: str, max_pages: int = 12, ua: str = None, use_js: bool = False) -> dict:
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

        final_url = url
        status = 0
        html = ""
        is_html = False
        ctype = ""
        try:
            final_url, status, html, is_html, ctype = _fetch(url, ua)
        except Exception:
            pass

        title = h1 = meta_desc = canon = None
        h2 = []; h3 = []; paragraphs = []
        noindex = nofollow = False
        issues = []
        og_title = og_desc = twitter_card = None
        jsonld_types = []
        jsonld_blocks = []
        faq_jsonld = []
        faq_visible = []
        outlinks = []
        word_count = 0
        html_out = ""

        if is_html and html:
            title = _extract_title(html)
            h1 = _extract_tag(html, "h1")
            h2 = _extract_multi(html, "h2", maxn=10)
            h3 = _extract_multi(html, "h3", maxn=10)
            meta_desc = _extract_meta(html, "description")
            og_title = _extract_meta_property(html, "og:title")
            og_desc  = _extract_meta_property(html, "og:description")
            twitter_card = _extract_meta_property(html, "twitter:card") or _extract_meta(html, "twitter:card")
            canon = _canonical(html)
            noindex, nofollow = _robots_meta(html)
            paragraphs = _extract_paragraphs(html, maxn=3, max_chars=500)
            jsonld_types = _extract_jsonld_types(html)
            jsonld_blocks = _extract_jsonld_blocks(html)
            for b in jsonld_blocks:
                faq_jsonld.extend(_collect_faq_from_jsonld(b))
            faq_visible = _extract_visible_faq(html, maxn=30)
            outlinks = [l for l in _extract_links(html, final_url) if _same_host(start_host, l)]
            word_count = _visible_text_word_count(html)
            html_out = html[:MAX_HTML_BYTES]

            if not meta_desc: issues.append("missing_meta_description")
            if not h1: issues.append("missing_h1")
            if canon and _normalize_host(canon.rstrip("/")) != _normalize_host(final_url.rstrip("/")):
                issues.append("canonical_differs")
            if not title: issues.append("missing_title")
            if "noindex" in issues: pass

        # JS-render fallback alleen als niets gevonden en toegestaan
        if use_js and is_html and not faq_jsonld and not faq_visible:
            try_html = _try_js_render(final_url, ua)
            if try_html:
                # re-extract alleen FAQ relevante dingen + essentials
                if not title: title = _extract_title(try_html)
                if not h1: h1 = _extract_tag(try_html, "h1")
                if not meta_desc: meta_desc = _extract_meta(try_html, "description")
                if not canon: canon = _canonical(try_html)
                if not og_title: og_title = _extract_meta_property(try_html, "og:title")
                if not og_desc:  og_desc  = _extract_meta_property(try_html, "og:description")
                if not twitter_card: twitter_card = _extract_meta_property(try_html, "twitter:card") or _extract_meta(try_html, "twitter:card")
                if not paragraphs: paragraphs = _extract_paragraphs(try_html, maxn=3, max_chars=500)
                if not jsonld_types: jsonld_types = _extract_jsonld_types(try_html)
                if not outlinks:
                    outlinks = [l for l in _extract_links(try_html, final_url) if _same_host(start_host, l)]
                if not word_count: word_count = _visible_text_word_count(try_html)

                blocks2 = _extract_jsonld_blocks(try_html)
                for b in blocks2:
                    faq_jsonld.extend(_collect_faq_from_jsonld(b))
                faq_visible2 = _extract_visible_faq(try_html, maxn=30)
                if faq_visible2: faq_visible = faq_visible2

                # combineer html (pref: rendered)
                html_out = try_html[:MAX_HTML_BYTES]

        pages.append({
            "url": url,
            "final_url": final_url,
            "status": status,
            "content_type": ctype,
            "title": title,
            "h1": h1,
            "h2": h2,
            "h3": h3,
            "meta_description": meta_desc,
            "og_title": og_title,
            "og_description": og_desc,
            "twitter_card": twitter_card,
            "canonical": canon,
            "noindex": noindex,
            "nofollow": nofollow,
            "paragraphs": paragraphs,
            "jsonld_types": jsonld_types,
            "faq_jsonld_count": len(faq_jsonld),
            "faq_visible_count": len(faq_visible),
            "faq_jsonld": faq_jsonld[:30],
            "faq_visible": faq_visible[:30],
            "html": html_out,
            "links": outlinks,
            "word_count": word_count,
            "issues": issues
        })

        html = None
        gc.collect()

        # queue interne links
        for link in outlinks:
            if link not in seen and not _seems_asset(link) and len(pages) + len(queue) < (max_pages * 3):
                queue.append(link)

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

def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("start_url", help="Start URL (incl. protocol)")
    ap.add_argument("--max-pages", type=int, default=12)
    ap.add_argument("--ua", type=str, default=DEFAULT_UA)
    ap.add_argument("--js", type=int, default=int(os.getenv("CRAWL_USE_JS", "0")), help="JS render fallback via Playwright (0/1)")
    ap.add_argument("--out", type=str, default="", help="Pad naar outputbestand; default stdout")
    args = ap.parse_args()

    res = crawl_site(args.start_url, max_pages=args.max_pages, ua=args.ua, use_js=bool(args.js))
    data = json.dumps(res, ensure_ascii=False, separators=(",", ":"), indent=2)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(data)
    else:
        sys.stdout.write(data)

if __name__ == "__main__":
    _cli()
