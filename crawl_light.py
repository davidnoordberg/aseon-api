# crawl_light.py — Robust crawler met volledige FAQ-extractie (DOM + JSON-LD)
# Public API: crawl_site(start_url: str, max_pages: int = 40, ua: Optional[str] = None) -> dict

import os
import re
import gc
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

DEFAULT_TIMEOUT = float(os.getenv("CRAWL_TIMEOUT_SEC", "10"))
MAX_PAGES = int(os.getenv("CRAWL_MAX_PAGES", "40"))
MAX_HTML_BYTES = int(os.getenv("CRAWL_MAX_HTML_BYTES", "1500000"))

_SKIP_EXT = {
    ".png",".jpg",".jpeg",".webp",".gif",".svg",".ico",".bmp",".avif",
    ".pdf",".zip",".rar",".7z",".gz",".mp4",".mp3",".mov",".wav",".woff",".woff2",".ttf",".eot",".otf",".webm"
}

QUESTION_PREFIXES = (
    "what ","how ","why ","when ","can ","does ","do ","is ","are ","should ","will ","where ","who ",
    "wat ","hoe ","waarom ","wanneer ","kan ","doet ","doen ","is ","zijn ","moet ","zal ","waar ","wie "
)

def _seems_asset(url: str) -> bool:
    try:
        path = urlparse(url).path.lower()
        ext = (path.rsplit(".", 1)[-1] if "." in path else "")
        return ("."+ext) in _SKIP_EXT
    except Exception:
        return False

def _norm_url(url: str) -> str:
    try:
        u = urlparse(url.strip())
        if not u.scheme:
            return ""
        netloc = u.netloc.lower()
        path = re.sub(r"/{2,}", "/", u.path or "/")
        return urlunparse((u.scheme, netloc, path, "", u.query, ""))
    except Exception:
        return ""

def _same_host(a: str, b: str) -> bool:
    try:
        return urlparse(a).netloc.lower() == urlparse(b).netloc.lower()
    except Exception:
        return False

def _clean(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s

def _text_of(node: Tag) -> str:
    if node is None:
        return ""
    if isinstance(node, NavigableString):
        return _clean(str(node))
    if not isinstance(node, Tag):
        return ""
    for bad in node.find_all(["script","style","noscript","template"]):
        bad.decompose()
    return _clean(node.get_text(separator=" "))

def _looks_like_question(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    low = t.lower()
    if "?" in t:
        return True
    if any(low.startswith(p) for p in QUESTION_PREFIXES):
        return True
    if re.match(r"^(q|vraag)\s*[:\-–]\s*\S", low):
        return True
    return False

def _is_empty_answer(text: str) -> bool:
    if not text: return True
    words = text.strip().split()
    return len(words) < 2

def _collect_dom_blocks(soup: BeautifulSoup) -> List[Dict[str, str]]:
    blocks: List[Dict[str,str]] = []
    TAGS = ["h1","h2","h3","h4","h5","h6","p","li","dt","dd","summary","button","a"]
    walker = soup.body or soup
    for el in walker.descendants:
        if isinstance(el, Tag) and el.name in TAGS:
            txt = _text_of(el)
            if txt:
                blocks.append({"tag": el.name, "text": txt})
    out: List[Dict[str,str]] = []
    for b in blocks:
        if out and out[-1]["tag"] == b["tag"] and out[-1]["text"] == b["text"]:
            continue
        out.append(b)
    return out

def _pair_dom_qas(dom_blocks: List[Dict[str,str]]) -> List[Dict[str,str]]:
    qas: List[Dict[str,str]] = []
    used_idx = set()
    N = len(dom_blocks)
    for i in range(N):
        blk = dom_blocks[i]
        qtxt = blk.get("text","")
        if not _looks_like_question(qtxt):
            continue
        ans_parts: List[str] = []
        for j in range(i+1, min(i+30, N)):
            if j in used_idx:
                continue
            cand = dom_blocks[j].get("text","")
            if _looks_like_question(cand):
                break
            if dom_blocks[j]["tag"] in ("h1","h2","h3","h4","h5","h6","dt","summary","button"):
                if not ans_parts:
                    continue
                else:
                    break
            if sum(len(x) for x in ans_parts)+len(cand) > 1200:
                break
            ans_parts.append(cand)
        ans = _clean(" ".join(ans_parts))
        if not _is_empty_answer(ans):
            used_idx.add(i)
            qas.append({"q": qtxt if qtxt.endswith("?") else (qtxt.rstrip(".! ")+"?"), "a": ans})
    return _dedupe_qas(qas)

def _dl_qas(soup: BeautifulSoup) -> List[Dict[str,str]]:
    out: List[Dict[str,str]] = []
    for dl in soup.find_all("dl"):
        dts = [el for el in dl.find_all("dt", recursive=False)]
        dds = [el for el in dl.find_all("dd", recursive=False)]
        if dts and dds:
            for i, dt in enumerate(dts):
                q = _text_of(dt)
                a = _text_of(dds[i]) if i < len(dds) else ""
                if _looks_like_question(q) and not _is_empty_answer(a):
                    out.append({"q": q if q.endswith("?") else (q.rstrip(".! ")+"?"), "a": a})
    return out

def _details_qas(soup: BeautifulSoup) -> List[Dict[str,str]]:
    out: List[Dict[str,str]] = []
    for det in soup.find_all("details"):
        q_el = det.find("summary")
        if q_el:
            q = _text_of(q_el)
            for s in det.find_all("summary"):
                s.decompose()
            a = _text_of(det)
            if _looks_like_question(q) and not _is_empty_answer(a):
                out.append({"q": q if q.endswith("?") else (q.rstrip(".! ")+"?"), "a": a})
    return out

def _aria_accordion_qas(soup: BeautifulSoup) -> List[Dict[str,str]]:
    out: List[Dict[str,str]] = []
    for btn in soup.find_all(["button","a","div","h3","h4","h5"], attrs={"aria-controls": True}):
        q = _text_of(btn)
        target_id = btn.get("aria-controls")
        a = ""
        if target_id:
            tgt = soup.find(id=target_id)
            if tgt:
                a = _text_of(tgt)
        if _looks_like_question(q) and not _is_empty_answer(a):
            out.append({"q": q if q.endswith("?") else (q.rstrip(".! ")+"?"), "a": a})
    return out

def _class_based_faq_qas(soup: BeautifulSoup) -> List[Dict[str,str]]:
    out: List[Dict[str,str]] = []
    FAQ_CLASS_HINTS = re.compile(r"(faq|accordion|question|qna|q-and-a)", re.I)
    for container in soup.find_all(attrs={"class": FAQ_CLASS_HINTS}):
        q_el = None
        for tag in ["h2","h3","h4","h5","button","summary"]:
            q_el = container.find(tag)
            if q_el:
                break
        if not q_el:
            continue
        q = _text_of(q_el)
        try:
            q_el.extract()
        except Exception:
            pass
        a = _text_of(container)
        if _looks_like_question(q) and not _is_empty_answer(a):
            out.append({"q": q if q.endswith("?") else (q.rstrip(".! ")+"?"), "a": a})
    return out

def _dedupe_qas(qas: List[Dict[str,str]]) -> List[Dict[str,str]]:
    seen = set()
    out: List[Dict[str,str]] = []
    for qa in qas:
        q = _clean(qa.get("q",""))
        a = _clean(qa.get("a",""))
        if not q or not a:
            continue
        key = (q.lower(), a[:160].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({"q": q, "a": a})
    return out

def _extract_jsonld(soup: BeautifulSoup) -> Tuple[List[Any], List[Any]]:
    raw_blocks: List[Any] = []
    faq_blocks: List[Any] = []
    for sc in soup.find_all("script", attrs={"type": re.compile(r"application/ld\+json", re.I)}):
        try:
            txt = sc.string or sc.get_text() or ""
            txt = txt.strip()
            if not txt:
                continue
            obj = json.loads(txt)
            raw_blocks.append(obj)
            def walk(o):
                if isinstance(o, dict):
                    t = str(o.get("@type","")).lower()
                    if t == "faqpage" or "mainEntity" in o or "@graph" in o:
                        faq_blocks.append(o)
                    for v in o.values():
                        walk(v)
                elif isinstance(o, list):
                    for v in o:
                        walk(v)
            walk(obj)
        except Exception:
            continue
    return raw_blocks, faq_blocks

def _meta(soup: BeautifulSoup, name: str) -> str:
    el = soup.find("meta", attrs={"name": name})
    return el.get("content","").strip() if el else ""

def _meta_prop(soup: BeautifulSoup, prop: str) -> str:
    el = soup.find("meta", attrs={"property": prop})
    return el.get("content","").strip() if el else ""

def _canonical(soup: BeautifulSoup) -> str:
    el = soup.find("link", rel=lambda v: v and "canonical" in [x.lower() for x in (v if isinstance(v, list) else [v])])
    href = el.get("href","").strip() if el else ""
    return href

def _robots_meta(soup: BeautifulSoup) -> Tuple[bool,bool]:
    el = soup.find("meta", attrs={"name": re.compile(r"robots", re.I)})
    if not el: return (False, False)
    content = (el.get("content") or "").lower()
    return ("noindex" in content, "nofollow" in content)

def _extract_faq_visible(soup: BeautifulSoup, dom_blocks: List[Dict[str,str]]) -> List[Dict[str,str]]:
    qas: List[Dict[str,str]] = []
    qas += _dl_qas(soup)
    qas += _details_qas(soup)
    qas += _aria_accordion_qas(soup)
    qas += _class_based_faq_qas(soup)
    if len(qas) < 5 and dom_blocks:
        qas += _pair_dom_qas(dom_blocks)
    return _dedupe_qas(qas)

def _extract_links(soup: BeautifulSoup, base: str) -> List[str]:
    out: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("javascript:","mailto:","#")):
            continue
        url = urljoin(base, href)
        url = _norm_url(url)
        if not url: continue
        if _seems_asset(url): continue
        out.append(url)
    seen = set(); deduped=[]
    for u in out:
        if u not in seen:
            seen.add(u); deduped.append(u)
    return deduped

def _fetch(url: str, ua: Optional[str]) -> Tuple[int, str, str, bool]:
    headers = {"User-Agent": ua or "AseonBot/0.5 (+https://aseon.ai)"}
    resp = requests.get(url, headers=headers, timeout=DEFAULT_TIMEOUT, allow_redirects=True)
    resp.raise_for_status()
    html = resp.text if isinstance(resp.text, str) else ""
    if len(html.encode("utf-8","ignore")) > MAX_HTML_BYTES:
        html = html[:MAX_HTML_BYTES]
    ctype = (resp.headers.get("content-type") or "").lower()
    is_html = "text/html" in ctype or "<html" in html.lower()
    return resp.status_code, html if is_html else "", "text/html" if is_html else ctype, is_html

def crawl_site(start_url: str, max_pages: int = MAX_PAGES, ua: Optional[str] = None) -> Dict[str, Any]:
    start_url = _norm_url(start_url)
    if not start_url:
        raise ValueError("Invalid start_url")
    host = urlparse(start_url).netloc.lower()

    visited = set()
    queue: List[str] = [start_url]
    pages: List[Dict[str, Any]] = []

    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        url = _norm_url(url)
        if not url or url in visited:
            continue
        if not _same_host(url, start_url):
            continue
        try:
            status, html, ctype, is_html = _fetch(url, ua)
        except Exception:
            visited.add(url)
            continue

        visited.add(url)

        soup = BeautifulSoup(html or "", "html.parser") if is_html else BeautifulSoup("", "html.parser")

        title = _clean(soup.title.get_text()) if soup.title else ""
        h1 = [_clean(el.get_text()) for el in soup.find_all("h1")]
        h2 = [_clean(el.get_text()) for el in soup.find_all("h2")]
        h3 = [_clean(el.get_text()) for el in soup.find_all("h3")]
        paragraphs = [_clean(el.get_text()) for el in soup.find_all("p")]
        li = [_clean(el.get_text()) for el in soup.find_all("li")]
        dt = [_clean(el.get_text()) for el in soup.find_all("dt")]
        dd = [_clean(el.get_text()) for el in soup.find_all("dd")]
        summary = [_clean(el.get_text()) for el in soup.find_all("summary")]
        buttons = [_clean(el.get_text()) for el in soup.find_all("button")]

        dom_blocks = _collect_dom_blocks(soup) if is_html else []

        raw_jsonld, faq_ld = _extract_jsonld(soup)
        has_faq_schema = bool(faq_ld)

        faq_visible = _extract_faq_visible(soup, dom_blocks) if is_html else []

        page = {
            "url": url,
            "status": status,
            "title": title,
            "h1": h1[0] if h1 else "",
            "h2": h2,
            "h3": h3,
            "paragraphs": [p for p in paragraphs if p],
            "li": [x for x in li if x],
            "dt": [x for x in dt if x],
            "dd": [x for x in dd if x],
            "summary": [x for x in summary if x],
            "buttons": [x for x in buttons if x],
            "dom_blocks": dom_blocks,
            "faq_visible": faq_visible,
            "faq_jsonld": faq_ld,
            "metrics": {
                "has_faq_schema": has_faq_schema
            },
            "meta": {
                "description": _meta(soup, "description"),
                "og:title": _meta_prop(soup, "og:title"),
                "og:description": _meta_prop(soup, "og:description"),
                "twitter:card": _meta_prop(soup, "twitter:card") or _meta(soup, "twitter:card"),
            },
            "canonical": _canonical(soup),
            "robots": {
                "noindex": _robots_meta(soup)[0],
                "nofollow": _robots_meta(soup)[1],
            }
        }

        pages.append(page)

        for link in _extract_links(soup, url):
            if link not in visited and link not in queue and _same_host(link, start_url) and not _seems_asset(link):
                queue.append(link)

        if len(pages) % 5 == 0:
            gc.collect()

    title_lengths = [len((p.get("title") or "")) for p in pages]
    missing_meta_desc = [p for p in pages if not (p.get("meta") or {}).get("description")]
    missing_h1 = [p for p in pages if not p.get("h1")]
    canonical_differs = [
        p for p in pages
        if p.get("canonical") and _norm_url(p["canonical"]) != p["url"]
    ]
    summary = {
        "page_count": len(pages),
        "avg_title_len": (sum(title_lengths)/len(title_lengths)) if title_lengths else 0,
        "faq_pages": sum(1 for p in pages if p.get("faq_visible") or p.get("faq_jsonld")),
        "has_faq_schema_count": sum(1 for p in pages if bool((p.get("metrics") or {}).get("has_faq_schema"))),
    }
    quick_wins = []
    if missing_meta_desc:
        quick_wins.append({"type": "missing_meta_description"})
    if missing_h1:
        quick_wins.append({"type": "missing_h1"})
    if canonical_differs:
        quick_wins.append({"type": "canonical_differs"})

    return {"start_url": start_url, "pages": pages, "summary": summary, "quick_wins": quick_wins}
