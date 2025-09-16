# report_agent.py

import os
import io
import re
import json
import csv
import base64
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit
from html import unescape as html_unescape

from psycopg.rows import dict_row

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
    XPreformatted,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from xml.sax.saxutils import escape as xml_escape

from openai import OpenAI

OPENAI_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "30"))
_openai_key = os.getenv("OPENAI_API_KEY")
_openai_client = OpenAI(api_key=_openai_key) if _openai_key else None


# -----------------------------------------------------
# DB helpers
# -----------------------------------------------------
def _fetch_latest_job(conn, site_id: str, jtype: str):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT output
              FROM jobs
             WHERE site_id=%s AND type=%s AND status='done'
             ORDER BY finished_at DESC NULLS LAST, created_at DESC
             LIMIT 1
            """,
            (site_id, jtype),
        )
        r = cur.fetchone()
        return (r or {}).get("output") if r else None


def _fetch_site_meta(conn, site_id: str) -> Dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT s.url, s.language, s.country, a.name AS account_name
              FROM sites s
              JOIN accounts a ON a.id = s.account_id
             WHERE s.id=%s
            """,
            (site_id,),
        )
        return cur.fetchone() or {}


# -----------------------------------------------------
# URL/normalize helpers
# -----------------------------------------------------
def _norm_url(u: str) -> str:
    if not u:
        return ""
    p = urlsplit(u)
    scheme = p.scheme or "https"
    host = p.hostname or ""
    path = (p.path or "/")
    if not path:
        path = "/"
    if not path.startswith("/"):
        path = "/" + path
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return urlunsplit((scheme, host, path, "", ""))


# -----------------------------------------------------
# PDF helpers
# -----------------------------------------------------
def _styles():
    styles = getSampleStyleSheet()
    if "Small" not in styles.byName:
        styles.add(ParagraphStyle(name="Small", fontSize=9, leading=12))
    if "Tiny" not in styles.byName:
        styles.add(ParagraphStyle(name="Tiny", fontSize=8, leading=10))
    if "MonoSmall" not in styles.byName:
        styles.add(ParagraphStyle(name="MonoSmall", fontName="Courier", fontSize=8, leading=10))
    if "H3tight" not in styles.byName:
        styles.add(ParagraphStyle(name="H3tight", parent=styles["Heading3"], spaceBefore=6, spaceAfter=4))
    return styles


def P(text: str, style_name: str = "Small") -> Paragraph:
    s = _styles()
    safe = xml_escape(str(text or "")).replace("\n", "<br/>")
    return Paragraph(safe, s[style_name])


def Code(text: str) -> XPreformatted:
    s = _styles()
    return XPreformatted(text or "", s["MonoSmall"])


def _make_table(data: List[List[Any]], col_widths: List[float]) -> Table:
    t = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F2F2F2")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C8C8C8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def _shorten(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    return (s[: max_chars - 1] + "…") if len(s) > max_chars else s


def _attr_escape(s: str) -> str:
    return xml_escape(s or "").replace('"', "&quot;")


# -----------------------------------------------------
# Language detection (light)
# -----------------------------------------------------
_NL_HINTS = [" de ", " het ", " een ", " en ", " voor ", " met ", " jouw ", " je ", " wij ", " onze "]
_EN_HINTS = [" the ", " and ", " for ", " with ", " your ", " we ", " our ", " to ", " of "]

def _detect_lang(texts: List[str], site_lang: Optional[str]) -> str:
    site = (site_lang or "").lower()
    default = "nl" if site.startswith("nl") else "en"
    sample = (" ".join([t for t in texts if t])[:800] + " ").lower()
    nl_score = sum(1 for w in _NL_HINTS if w in sample)
    en_score = sum(1 for w in _EN_HINTS if w in sample)
    if nl_score > en_score:
        return "nl"
    if en_score > nl_score:
        return "en"
    return default


# -----------------------------------------------------
# Core helpers (unique pages, duplicates, proposals)
# -----------------------------------------------------
def _unique_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bucket: Dict[str, Dict[str, Any]] = {}

    def _score(pp: Dict[str, Any]) -> Tuple[int, int]:
        return (1 if (pp.get("status") == 200) else 0, int(pp.get("word_count") or 0))

    for p in pages or []:
        u = _norm_url(p.get("final_url") or p.get("url") or "")
        if not u:
            continue
        cur = bucket.get(u)
        if (cur is None) or (_score(p) > _score(cur)):
            bucket[u] = p
    return [{**v, "final_url": u, "url": u} for u, v in bucket.items()]


def _dup_map(values: List[Optional[str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for v in values:
        if not v:
            continue
        k = v.strip().lower()
        if not k:
            continue
        counts[k] = counts.get(k, 0) + 1
    return counts


def _build_link_graph(pages: List[Dict[str, Any]]) -> Dict[str, int]:
    inlinks: Dict[str, int] = {}
    url_index = {}
    for p in pages:
        u = _norm_url(p.get("final_url") or p.get("url") or "")
        url_index[u] = True
    for p in pages:
        for l in (p.get("links") or []):
            tgt = _norm_url(l)
            if tgt in url_index:
                inlinks[tgt] = inlinks.get(tgt, 0) + 1
    return inlinks


# -----------------------------------------------------
# Copy proposals via LLM or fallback
# -----------------------------------------------------
def _propose_copy(
    url: str,
    title: Optional[str],
    h1: Optional[str],
    meta: Optional[str],
    h2_list: Optional[List[str]],
    h3_list: Optional[List[str]],
    paras: Optional[List[str]],
    lang: str = "en",
) -> Dict[str, str]:
    base = (h1 or title or "").strip()

    if lang == "nl":
        fallback = {
            "title": (base[:60] if base else "Startpagina"),
            "meta": "Korte, duidelijke samenvatting van de pagina (140–155 tekens) die uitnodigt tot doorklikken.",
            "h1_alt": (h1 or title or "Welkom"),
            "intro": "Antwoord in 1–2 zinnen wat de bezoeker hier kan doen en waarom dit relevant is.",
        }
    else:
        fallback = {
            "title": (base[:60] if base else "Homepage"),
            "meta": "A short, clear summary of the page (140–155 chars) that invites the click.",
            "h1_alt": (h1 or title or "Welcome"),
            "intro": "In 1–2 sentences, state what the visitor can do here and why it matters.",
        }

    if not _openai_client:
        out = fallback
    else:
        ctx_bits: List[str] = []
        if h1:
            ctx_bits.append(f"H1: {h1}")
        if title:
            ctx_bits.append(f"Title: {title}")
        if meta:
            ctx_bits.append(f"Meta: {meta}")
        for h in (h2_list or [])[:3]:
            ctx_bits.append(f"H2: {h}")
        for h in (h3_list or [])[:2]:
            ctx_bits.append(f"H3: {h}")
        for p in (paras or [])[:2]:
            ctx_bits.append(f"P: {p}")

        if lang == "nl":
            sys = (
                "Je schrijft compacte webkopie in het Nederlands. "
                "Lever een <title> (≤60 tekens) en meta description (140–155 tekens), "
                "een H1-variant (niet identiek aan de title) en een korte intro (1–2 zinnen). "
                "Retourneer JSON met velden: title, meta, h1_alt, intro."
            )
        else:
            sys = (
                "You write compact web copy in English. "
                "Provide a <title> (≤60 chars), a meta description (140–155 chars), "
                "an H1 variant (not identical to the title) and a short intro (1–2 sentences). "
                "Return JSON fields: title, meta, h1_alt, intro."
            )

        user = f"URL: {url}\nCONTEXT:\n" + "\n".join(ctx_bits)[:1200]

        try:
            resp = _openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.2,
                timeout=OPENAI_TIMEOUT_SEC,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            )
            data = json.loads(resp.choices[0].message.content)
            out = {
                "title": (data.get("title") or fallback["title"]).strip()[:60],
                "meta": re.sub(r"\s+", " ", (data.get("meta") or fallback["meta"]).strip())[:160],
                "h1_alt": (data.get("h1_alt") or fallback["h1_alt"]).strip(),
                "intro": (data.get("intro") or fallback["intro"]).strip(),
            }
        except Exception:
            out = fallback

    og_title = out["title"]
    og_desc = out["meta"]
    return {**out, "og_title": og_title, "og_desc": og_desc}


# -----------------------------------------------------
# Text issue builder
# -----------------------------------------------------
_TEXT_FINDINGS_ALL = {
    "Missing <title>",
    "Title length suboptimal",
    "Duplicate <title>",
    "Missing meta description",
    "Meta description length suboptimal",
    "Duplicate meta description",
    "H1 duplicates title",
    "Missing H1",
    "Low visible text",
}

_PATCHABLE_FINDINGS = {"Missing canonical", "Canonical differs from page URL", "Missing Open Graph tags"}

def _build_text_findings(crawl: Optional[Dict[str, Any]], site_lang: Optional[str]) -> List[Dict[str, Any]]:
    if not crawl:
        return []
    pages = _unique_pages(crawl.get("pages") or [])

    title_counts = _dup_map([p.get("title") for p in pages if p.get("status") == 200])
    meta_counts = _dup_map([p.get("meta_description") for p in pages if p.get("status") == 200])

    rows: List[Dict[str, Any]] = []
    for p in pages:
        url = _norm_url(p.get("final_url") or p.get("url") or "")
        title = (p.get("title") or "").strip()
        h1 = (p.get("h1") or "").strip()
        meta = (p.get("meta_description") or "").strip()
        h2_list = p.get("h2") or []
        h3_list = p.get("h3") or []
        paras = p.get("paragraphs") or []
        wc = int(p.get("word_count") or 0)

        lang = _detect_lang([title, h1, meta] + paras, site_lang)

        need_title = (not title) or len(title) < 30 or len(title) > 65 or title_counts.get(title.lower(), 0) > 1
        need_meta = (not meta) or len(meta) < 70 or len(meta) > 180 or meta_counts.get(meta.lower(), 0) > 1
        h1_eq_title = bool(title and h1 and h1.strip().lower() == title.strip().lower())
        low_text = (wc and wc < 250)

        if not any([need_title, need_meta, h1_eq_title, low_text]):
            continue

        prop = _propose_copy(url, title, h1, meta, h2_list, h3_list, paras, lang=lang)

        if need_title:
            reason = []
            if not title:
                reason.append("missing" if lang != "nl" else "ontbreekt")
            if title and (len(title) < 30 or len(title) > 65):
                reason.append("length suboptimal" if lang != "nl" else "lengte suboptimaal")
            if title and title_counts.get(title.lower(), 0) > 1:
                reason.append("duplicate site-wide" if lang != "nl" else "dubbel op site")
            rows.append(
                {
                    "url": url,
                    "field": "title",
                    "problem": ", ".join(reason),
                    "current": title or ("(empty)" if lang != "nl" else "(leeg)"),
                    "proposed": prop["title"],
                    "html_patch": f"<title>{xml_escape(prop['title'])}</title>",
                    "lang": lang,
                }
            )

        if need_meta:
            reason = []
            if not meta:
                reason.append("missing" if lang != "nl" else "ontbreekt")
            if meta and (len(meta) < 140 or len(meta) > 160):
                reason.append("length suboptimal" if lang != "nl" else "lengte suboptimaal")
            if meta and meta_counts.get(meta.lower(), 0) > 1:
                reason.append("duplicate site-wide" if lang != "nl" else "dubbel op site")
            md = prop["meta"]
            rows.append(
                {
                    "url": url,
                    "field": "meta_description",
                    "problem": ", ".join(reason),
                    "current": meta or ("(empty)" if lang != "nl" else "(leeg)"),
                    "proposed": md,
                    "html_patch": f"<meta name=\"description\" content=\"{_attr_escape(md)}\">",
                    "lang": lang,
                }
            )

        if h1_eq_title:
            h1_alt = prop["h1_alt"]
            rows.append(
                {
                    "url": url,
                    "field": "h1",
                    "problem": "H1 duplicates title" if lang != "nl" else "H1 is identiek aan title",
                    "current": h1,
                    "proposed": h1_alt,
                    "html_patch": f"<h1>{xml_escape(h1_alt)}</h1>",
                    "lang": lang,
                }
            )

        if low_text:
            intro = prop["intro"]
            rows.append(
                {
                    "url": url,
                    "field": "intro_paragraph",
                    "problem": "low visible text (<250 words)" if lang != "nl" else "weinig zichtbare tekst (<250 woorden)",
                    "current": (paras[0] if paras else ("(no visible paragraph)" if lang != "nl" else "(geen tekst)")),
                    "proposed": intro,
                    "html_patch": f"<p>{xml_escape(intro)}</p>",
                    "lang": lang,
                }
            )

    return rows


# -----------------------------------------------------
# Duplicates section
# -----------------------------------------------------
def _build_duplicate_groups(crawl: Optional[Dict[str, Any]], site_lang: Optional[str]):
    if not crawl:
        return {"title_groups": [], "meta_groups": []}

    pages = _unique_pages(crawl.get("pages") or [])
    title_buckets: Dict[str, List[Dict[str, Any]]] = {}
    meta_buckets: Dict[str, List[Dict[str, Any]]] = {}

    for p in pages:
        title = (p.get("title") or "").strip()
        meta = (p.get("meta_description") or "").strip()
        if title:
            key = title.lower()
            title_buckets.setdefault(key, []).append(p)
        if meta:
            key = meta.lower()
            meta_buckets.setdefault(key, []).append(p)

    title_groups, meta_groups = [], []

    for _, plist in title_buckets.items():
        if len(plist) <= 1:
            continue
        group_rows = []
        for p in plist:
            url = _norm_url(p.get("final_url") or p.get("url") or "")
            lang = _detect_lang([p.get("title") or "", p.get("h1") or ""], site_lang)
            prop = _propose_copy(
                url,
                p.get("title"),
                p.get("h1"),
                p.get("meta_description"),
                p.get("h2"),
                p.get("h3"),
                p.get("paragraphs"),
                lang=lang,
            )
            group_rows.append({"url": url, "current": p.get("title") or "", "proposed": prop["title"], "lang": lang})
        title_groups.append({"value": plist[0].get("title") or "", "count": len(plist), "rows": group_rows})

    for _, plist in meta_buckets.items():
        if len(plist) <= 1:
            continue
        group_rows = []
        for p in plist:
            url = _norm_url(p.get("final_url") or p.get("url") or "")
            lang = _detect_lang([p.get("meta_description") or "", p.get("title") or ""], site_lang)
            prop = _propose_copy(
                url,
                p.get("title"),
                p.get("h1"),
                p.get("meta_description"),
                p.get("h2"),
                p.get("h3"),
                p.get("paragraphs"),
                lang=lang,
            )
            group_rows.append({"url": url, "current": p.get("meta_description") or "", "proposed": prop["meta"], "lang": lang})
        meta_groups.append({"value": plist[0].get("meta_description") or "", "count": len(plist), "rows": group_rows})

    return {"title_groups": title_groups, "meta_groups": meta_groups}


# -----------------------------------------------------
# Technical SEO findings (deduped pages)
# -----------------------------------------------------
def _build_seo_findings(crawl: Optional[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not crawl:
        return [], []
    pages = _unique_pages(crawl.get("pages") or [])
    inlinks = _build_link_graph(pages)
    title_counts = _dup_map([p.get("title") for p in pages if p.get("status") == 200])
    meta_counts = _dup_map([p.get("meta_description") for p in pages if p.get("status") == 200])

    out: List[Dict[str, Any]] = []
    for p in pages:
        u = _norm_url(p.get("final_url") or p.get("url") or "")
        status = int(p.get("status") or 0)
        title = p.get("title")
        h1 = p.get("h1")
        md = p.get("meta_description")
        canon = (p.get("canonical") or "").strip()
        noindex = bool(p.get("noindex"))
        og_title = p.get("og_title")
        og_desc = p.get("og_description")
        word_count = int(p.get("word_count") or 0)
        inl = inlinks.get(u, 0)

        def add(find, sev, fix, accept):
            out.append({"url": u, "finding": find, "severity": sev, "fix": fix, "accept": accept})

        if status >= 400:
            add("HTTP error " + str(status), "high", "Fix the endpoint to return 200 or remove from internal linking.", ["URL returns 200", "Internal links updated"])

        if not title:
            add("Missing <title>", "high", "Add a descriptive, unique <title> (30–65 chars) that matches page intent.", ["<title> present", "30–65 chars", "Unique site-wide"])
        else:
            if len(title) < 30 or len(title) > 65:
                add("Title length suboptimal", "medium", "Keep <title> between ~30–65 chars; lead with primary intent/entity.", ["30–65 chars", "Primary intent first"])
            if title_counts.get((title or "").strip().lower(), 0) > 1:
                add("Duplicate <title>", "high", "Make the title unique per page; differentiate with intent or entities.", ["Titles unique site-wide"])

        if not md:
            add("Missing meta description", "medium", "Add unique <meta name='description'> (~140–160 chars) that summarizes the answer/value.", ["Tag present", "140–160 chars", "Unique on site"])
        else:
            if len(md) < 70 or len(md) > 180:
                add("Meta description length suboptimal", "low", "Aim for ~140–160 chars; avoid truncation and thin summaries.", ["~140–160 chars"])
            if meta_counts.get((md or "").strip().lower(), 0) > 1:
                add("Duplicate meta description", "medium", "Rewrite to reflect page’s unique value; avoid reuse.", ["Descriptions unique site-wide"])

        if not h1:
            add("Missing H1", "high", "Add exactly one <h1> with the primary topic; keep 30–65 chars.", ["Exactly one <h1>", "Matches intent"])
        elif title and h1.strip().lower() == title.strip().lower():
            add("H1 duplicates title", "low", "Differentiate H1 slightly to add context; keep user-first phrasing.", ["H1 present", "Not identical to title"])

        if noindex:
            add("noindex set", "high", "Remove noindex for indexable pages; keep only on intentional pages (thank-you, internal tools).", ["No 'noindex' on valuable pages"])

        if not canon:
            add("Missing canonical", "low", "Add <link rel='canonical'> to the preferred URL to prevent duplicates.", ["Canonical present", "Equals preferred URL"])
        else:
            if _norm_url(canon) != u:
                add("Canonical differs from page URL", "medium", "Canonical should match the preferred URL; align host & path.", ["Canonical equals preferred URL"])

        if word_count and word_count < 250:
            add("Low visible text", "medium", "Add concise, answer-first copy and supporting evidence; target ≥250–400 words for key pages.", ["≥250 words visible", "Lead with answer"])

        if not og_title or not og_desc:
            add("Missing Open Graph tags", "low", "Add og:title and og:description for clean sharing/AI snippets.", ["og:title present", "og:description present"])

        if inl == 0:
            add("Orphan/low internal links", "medium", "Link to this page from at least 2 relevant pages using natural anchor text.", ["≥2 internal inlinks", "Anchors descriptive"])

    return out, pages


# -----------------------------------------------------
# AEO helpers (normalize, parse, heuristics)
# -----------------------------------------------------
_FAQ_URL_HINTS = ("/faq", "/faqs", "/veelgestelde-vragen", "#faq")

def _is_likely_faq_url(u: str) -> bool:
    p = urlsplit(u)
    path = (p.path or "").lower()
    frag = (p.fragment or "").lower()
    joined = path + ("#" + frag if frag else "")
    return any(h in joined for h in _FAQ_URL_HINTS)

_TAG_RE = re.compile(r"<[^>]+>")

def _strip_html(s: str) -> str:
    if not s:
        return ""
    s = html_unescape(s)
    s = _TAG_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _trim_words(s: str, limit: int = 80) -> Tuple[str, bool]:
    words = (s or "").split()
    if len(words) <= limit:
        return " ".join(words), False
    return " ".join(words[:limit]) + "…", True

def _qas_from_jsonld(faq_jsonld: Any) -> List[Dict[str, str]]:
    """
    Extract Q/A pairs from a FAQPage JSON-LD object.
    Handles dict, list, minimal structure.
    """
    out: List[Dict[str, str]] = []

    def _handle_entity(entity):
        if not isinstance(entity, dict):
            return
        q = entity.get("name") or entity.get("question") or ""
        ans = entity.get("acceptedAnswer") or entity.get("accepted_answer")
        a_text = ""
        if isinstance(ans, list) and ans:
            ans = ans[0]
        if isinstance(ans, dict):
            a_text = ans.get("text") or ans.get("answer") or ""
        q = _strip_html(q)
        a_text = _strip_html(a_text)
        if q and a_text:
            out.append({"q": q, "a": a_text})

    def _handle_root(obj):
        if not isinstance(obj, dict):
            return
        # If it's a graph array wrapped in @graph
        if "@graph" in obj and isinstance(obj["@graph"], list):
            for node in obj["@graph"]:
                if isinstance(node, dict) and str(node.get("@type", "")).lower() == "faqpage":
                    main = node.get("mainEntity") or node.get("main_entity") or []
                    if isinstance(main, list):
                        for ent in main:
                            _handle_entity(ent)
                    else:
                        _handle_entity(main)
        # Plain FAQPage object
        main = obj.get("mainEntity") or obj.get("main_entity") or []
        if isinstance(main, list):
            for ent in main:
                _handle_entity(ent)
        else:
            _handle_entity(main)

    if isinstance(faq_jsonld, list):
        for item in faq_jsonld:
            if isinstance(item, dict):
                t = str(item.get("@type", "")).lower()
                if t == "faqpage" or "mainEntity" in item or "@graph" in item:
                    _handle_root(item)
    elif isinstance(faq_jsonld, dict):
        _handle_root(faq_jsonld)

    # Dedup by normalized question
    seen = set()
    deduped: List[Dict[str, str]] = []
    for qa in out:
        key = qa["q"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(qa)
    return deduped

def _infer_ptype(url: str, page: Dict[str, Any]) -> str:
    given = (page.get("type") or "").lower().strip()
    if given == "faq":
        return "faq"
    # Heuristics
    if page.get("faq_jsonld"):
        return "faq"
    metrics = page.get("metrics") or {}
    if bool(metrics.get("has_faq_schema")):
        return "faq"
    if _is_likely_faq_url(url):
        return "faq"
    return "other"


# -----------------------------------------------------
# AEO (quality heuristics from FAQ job)
# -----------------------------------------------------
def _aeo_findings_from_faq(faq: Optional[Dict[str, Any]], crawl: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if faq:
        for item in (faq.get("faqs") or []):
            q = (item.get("q") or "").strip()
            a = (item.get("a") or "").strip()
            src = item.get("source")
            if not src:
                out.append(
                    {
                        "url": (faq.get("site") or {}).get("url") or "",
                        "finding": f"FAQ missing source link — “{_shorten(q, 60)}”",
                        "severity": "medium",
                        "fix": "Add a canonical source URL per FAQ so assistants can verify and cite.",
                        "accept": ["Each FAQ has a working source URL", "Source resolves with HTTP 200"],
                    }
                )
            if len(a.split()) > 90:
                out.append(
                    {
                        "url": (faq.get("site") or {}).get("url") or "",
                        "finding": f"FAQ answer too long — “{_shorten(q, 60)}”",
                        "severity": "low",
                        "fix": "Trim to ≤80 words; lead with the answer; keep nouns/verbs concrete.",
                        "accept": ["≤80 words", "Lead sentence answers the question"],
                    }
                )
    has_faq = False
    if crawl:
        for p in (crawl.get("pages") or []):
            types = [t.lower() for t in (p.get("jsonld_types") or [])]
            if any(t == "faqpage" for t in types):
                has_faq = True
                break
    if not has_faq:
        out.append(
            {
                "url": (crawl or {}).get("start_url") or "",
                "finding": "No FAQPage schema detected",
                "severity": "medium",
                "fix": "Add a compact FAQ block on the dedicated FAQ page and include matching FAQPage JSON-LD.",
                "accept": ["Rich Results Test: valid FAQPage", "Answers ≤80 words", "JSON-LD matches visible Q&A on the FAQ page"],
            }
        )
    return out


# -----------------------------------------------------
# AEO (from AEO job: scorecards/issues -> findings)
# -----------------------------------------------------
def _aeo_findings_from_aeo_job(aeo_concrete: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not aeo_concrete:
        return out
    for page in (aeo_concrete.get("scorecards") or []):
        url = page.get("url") or ""
        ptype = (page.get("type") or "other").lower()
        metrics = page.get("metrics") or {}
        qas_detected = int(metrics.get("qas_detected") or metrics.get("qas") or 0)
        has_faq = bool(metrics.get("has_faq_schema_detected") or metrics.get("has_faq_schema"))

        if ptype == "faq":
            if qas_detected == 0:
                out.append({
                    "url": url,
                    "finding": "No Q&A section on page",
                    "severity": "high",
                    "fix": "Add a short FAQ section (3–6 Q&A) focused on the page’s main intents.",
                    "accept": ["FAQ block visible", "≥3 Q&A present", "Answers ≤80 words"],
                })
            elif qas_detected < 3:
                out.append({
                    "url": url,
                    "finding": "Too few Q&A on page",
                    "severity": "medium",
                    "fix": "Add 3–6 concise Q&A with answers ≤80 words; cover the core intents of the page.",
                    "accept": ["≥3 Q&A present", "Each answer ≤80 words", "Lead sentence contains the answer"],
                })
            if not has_faq:
                out.append({
                    "url": url,
                    "finding": "No FAQPage JSON-LD",
                    "severity": "medium",
                    "fix": "Add valid FAQPage JSON-LD matching on-page Q&A.",
                    "accept": ["Rich Results Test: valid FAQPage", "JSON-LD matches visible questions/answers"],
                })
            if int(metrics.get("answers_gt_80w") or 0) > 0:
                out.append({
                    "url": url,
                    "finding": "Overlong FAQ answers",
                    "severity": "low",
                    "fix": "Trim answers to ≤80 words; lead with the answer and keep nouns/verbs concrete.",
                    "accept": ["All answers ≤80 words", "Answer appears in sentence 1"],
                })
    return out


# -----------------------------------------------------
# AEO concrete (from aeo job) -> scorecards, Q&A rows, patches, plan_items
# -----------------------------------------------------
def _task_texts_for_content(field: str, lang: str) -> Tuple[str, str]:
    nl = lang == "nl"
    f = (field or "").lower()
    mapping = {
        "hero": ("Hero section", "Answer-first headline + subhead + primary CTA."),
        "subhead": ("Subhead", "Clarify value in one sentence."),
        "value_props": ("Value props", "3–4 concrete benefits in bullets."),
        "steps": ("Process steps", "3–5 simple steps to success."),
        "proof": ("Social proof", "Logos, quotes, stats to build trust."),
        "ctas": ("Calls to action", "Primary/secondary next steps."),
        "intro_paragraph": ("Intro paragraph", "Short, answer-first intro."),
    }
    if nl:
        mapping = {
            "hero": ("Hero-sectie", "Antwoord-first kop + subhead + primaire CTA."),
            "subhead": ("Subhead", "Verduidelijk de waarde in één zin."),
            "value_props": ("Value props", "3–4 concrete voordelen in bullets."),
            "steps": ("Stappenplan", "3–5 eenvoudige stappen naar resultaat."),
            "proof": ("Social proof", "Logo’s, quotes, stats voor vertrouwen."),
            "ctas": ("CTA’s", "Primaire/secundaire vervolgstappen."),
            "intro_paragraph": ("Intro-alinea", "Korte, antwoord-first intro."),
        }
    label, why = mapping.get(f, (("Content patch", "Improve clarity and conversion.") if not nl else ("Content patch", "Verbeter duidelijkheid en conversie.")))
    return label, why


def _aeo_from_job(aeo: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    patches: List[Dict[str, Any]] = []
    plan_items: List[Dict[str, Any]] = []
    scorecards: List[Dict[str, Any]] = []

    if not aeo:
        return {"rows": rows, "patches": patches, "plan_items": plan_items, "scorecards": scorecards}

    for page in (aeo.get("pages") or []):
        url = _norm_url(page.get("url") or "")
        lang = page.get("lang") or "en"
        ptype = _infer_ptype(url, page)

        qas_job = page.get("qas") or []
        qas_ld = _qas_from_jsonld(page.get("faq_jsonld") or {})
        merged_qas: List[Dict[str, str]] = []

        seen = set()
        for src in (qas_job, qas_ld):
            for qa in src:
                q = _strip_html(qa.get("q") or qa.get("question") or "")
                a = _strip_html(qa.get("a") or qa.get("answer") or "")
                if not q or not a:
                    continue
                k = q.strip().lower()
                if k in seen:
                    continue
                seen.add(k)
                merged_qas.append({"q": q, "a": a})

        answers_gt_80w = sum(1 for qa in merged_qas if len((qa["a"] or "").split()) > 80)

        if ptype == "faq" and merged_qas:
            for qa in merged_qas:
                trimmed, _ = _trim_words(qa["a"], 80)
                rows.append({"url": url, "q": qa["q"], "a": trimmed, "gaps": ", ".join(page.get("issues") or []) or "OK"})

        faq_html = page.get("faq_html") or ""
        faq_jsonld = page.get("faq_jsonld") or {}

        if ptype == "faq" and faq_html.strip():
            patches.append({
                "url": url, "field": "faq_html_block",
                "problem": "Missing short, snippet-ready Q&A on page",
                "current": "(none or not snippet-ready)",
                "proposed": "Add FAQ HTML block (3–6 Q&A, answers ≤80 words)",
                "html_patch": faq_html,
                "category": "body", "severity": 2, "impact": 5, "effort": 2, "priority": 6.0, "patchable": True
            })
            plan_items.append({
                "url": url,
                "task": "Add FAQ section (3–6 Q&A, ≤80 words)" if lang == "en" else "FAQ-blok toevoegen (3–6 Q&A, ≤80 woorden)",
                "why": "Improves answer/snippet eligibility in AE/LLM surfaces" if lang == "en" else "Verbetert snippet-eligibility in AE/LLM-oppervlakken",
                "category": "content",
                "field": "faq_html_block",
                "severity": 2, "impact": 5, "effort": 2, "effort_label": "M",
                "priority": 6.0,
                "patchable": True,
                "html_patch": faq_html,
                "lang": lang
            })

        if ptype == "faq" and faq_jsonld:
            html = "<script type=\"application/ld+json\">" + json.dumps(faq_jsonld, ensure_ascii=False) + "</script>"
            patches.append({
                "url": url, "field": "faq_jsonld",
                "problem": "No/invalid FAQPage JSON-LD",
                "current": "(none)",
                "proposed": "Inject <script type='application/ld+json'>FAQPage…</script> in <head> or <body>",
                "html_patch": html,
                "category": "head", "severity": 2, "impact": 4, "effort": 1, "priority": 6.0, "patchable": True
            })
            plan_items.append({
                "url": url,
                "task": "Add FAQPage JSON-LD" if lang == "en" else "FAQPage JSON-LD toevoegen",
                "why": "Validates Q&A for rich results; clearer AE extraction" if lang == "en" else "Valideert Q&A voor rich results; duidelijkere AE-extractie",
                "category": "tag",
                "field": "faq_jsonld",
                "severity": 2, "impact": 4, "effort": 1, "effort_label": "S",
                "priority": 6.0,
                "patchable": True,
                "html_patch": html,
                "lang": lang
            })

        for cp in (page.get("content_patches") or []):
            field = cp.get("field") or "content"
            html_patch = cp.get("html_patch") or ""
            category = cp.get("category") or "body"
            severity = int(cp.get("severity") or 2)
            impact = int(cp.get("impact") or 4)
            effort = int(cp.get("effort") or 2)
            priority = float(cp.get("priority") or 5.0)
            patchable = bool(cp.get("patchable", True))
            problem = cp.get("problem") or ("Missing content block" if lang == "en" else "Ontbrekend contentblok")
            proposed = cp.get("proposed") or None

            patches.append({
                "url": url,
                "field": field,
                "problem": problem,
                "current": cp.get("current") or "(none)",
                "proposed": proposed,
                "html_patch": html_patch,
                "category": category,
                "severity": severity,
                "impact": impact,
                "effort": effort,
                "priority": priority,
                "patchable": patchable,
            })

            task, why = _task_texts_for_content(field, lang)
            plan_items.append({
                "url": url,
                "task": task,
                "why": why,
                "category": "content",
                "field": field,
                "severity": severity,
                "impact": impact,
                "effort": effort,
                "effort_label": {1: "S", 2: "M", 3: "L"}.get(effort, "M"),
                "priority": priority,
                "patchable": patchable,
                "html_patch": html_patch,
                "lang": lang
            })

        score = int(page.get("score") or 0)
        issues = [str(i) for i in (page.get("issues") or [])]
        if ptype == "faq":
            if merged_qas and len(merged_qas) >= 3:
                issues = [i for i in issues if "too few q&a" not in i.lower()]
            if page.get("faq_jsonld") or (page.get("metrics") or {}).get("has_faq_schema"):
                issues = [i for i in issues if "no faqpage json-ld" not in i.lower()]

        metrics_in = page.get("metrics") or {}
        metrics = dict(metrics_in)
        metrics["qas_detected"] = len(merged_qas)
        metrics["answers_gt_80w"] = answers_gt_80w
        metrics["answers_leq_80w"] = max(0, len(merged_qas) - answers_gt_80w)
        metrics["has_faq_schema_detected"] = bool(page.get("faq_jsonld") or metrics_in.get("has_faq_schema"))

        scorecards.append({"url": url, "type": ptype, "score": score, "issues": issues, "metrics": metrics})

    return {"rows": rows, "patches": patches, "plan_items": plan_items, "scorecards": scorecards}


# -----------------------------------------------------
# GEO recommendations (entity/schema)
# -----------------------------------------------------
def _geo_recommendations(site_meta: Dict[str, Any], crawl: Optional[Dict[str, Any]], schema_job: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    home = (site_meta.get("url") or "").strip() or "https://example.com/"
    present_types = set()
    if crawl:
        for p in (crawl.get("pages") or []):
            for t in (p.get("jsonld_types") or []):
                present_types.add(t.lower())

    recs: List[Dict[str, Any]] = []
    if "organization" not in present_types or "website" not in present_types:
        recs.append(
            {
                "url": home,
                "finding": "Add/validate Organization + WebSite (entity grounding)",
                "severity": "high",
                "fix": "Provide Organization/Logo and WebSite JSON-LD; add sameAs to authoritative IDs (LinkedIn, Wikidata).",
                "accept": ["Rich Results Test: valid Organization & WebSite", "Logo URL HTTPS and public", "≥2 sameAs profiles resolve"],
            }
        )
    if "breadcrumblist" not in present_types:
        recs.append(
            {
                "url": home.rstrip("/") + "/faq",
                "finding": "Add BreadcrumbList on sections/FAQ",
                "severity": "medium",
                "fix": "Provide BreadcrumbList JSON-LD reflecting on-page breadcrumbs.",
                "accept": ["Valid BreadcrumbList", "Positions sequential (1..n)", "URLs canonical"],
            }
        )
    if schema_job and schema_job.get("biz_type", "").lower() == "faqpage":
        recs.append(
            {
                "url": home,
                "finding": "Broaden schema coverage beyond FAQPage",
                "severity": "medium",
                "fix": "Add Article/BlogPosting where applicable; add Person for authors; link entities with about/mentions.",
                "accept": ["Article/BlogPosting valid on content pages", "Author (Person) present with sameAs where appropriate"],
            }
        )
    return recs


# -----------------------------------------------------
# Canonical & OG patches
# -----------------------------------------------------
def _collect_proposals_by_url(text_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for r in text_rows:
        if r["field"] not in ("title", "meta_description"):
            continue
        u = r["url"]
        out.setdefault(u, {})
        if r["field"] == "title":
            out[u]["title"] = r["proposed"]
        if r["field"] == "meta_description":
            out[u]["meta"] = r["proposed"]
    for u, m in out.items():
        m.setdefault("title", None)
        m.setdefault("meta", None)
        if m["title"]:
            m["og_title"] = m["title"]
        if m["meta"]:
            m["og_desc"] = m["meta"]
    return out


def _build_canonical_og_patches(crawl: Optional[Dict[str, Any]], text_rows: List[Dict[str, Any]]):
    if not crawl:
        return []

    pages = _unique_pages(crawl.get("pages") or [])
    by_url = {_norm_url(p.get("final_url") or p.get("url") or ""): p for p in pages}
    proposals = _collect_proposals_by_url(text_rows)

    patches: List[Dict[str, Any]] = []
    for u, p in by_url.items():
        canon = (p.get("canonical") or "").strip()
        og_title = p.get("og_title")
        og_desc = p.get("og_description")

        preferred = u
        if not canon or _norm_url(canon) != preferred:
            patches.append(
                {
                    "url": u,
                    "category": "canonical",
                    "problem": "missing" if not canon else "differs from page URL",
                    "current": canon or "(none)",
                    "html_patch": f"<link rel=\"canonical\" href=\"{_attr_escape(preferred)}\">",
                }
            )

        ogt = proposals.get(u, {}).get("og_title") or (p.get("title") or "")[:60]
        ogd = proposals.get(u, {}).get("og_desc") or (p.get("meta_description") or "")[:200]

        if not og_title or not og_desc:
            patches.append(
                {
                    "url": u,
                    "category": "open_graph",
                    "problem": "missing og:title/og:description" if (not og_title and not og_desc) else ("missing og:title" if not og_title else "missing og:description"),
                    "current": f"og:title={'present' if og_title else 'missing'}, og:description={'present' if og_desc else 'missing'}",
                    "html_patch": (f"<meta property=\"og:title\" content=\"{_attr_escape(ogt)}\">\n"
                                   f"<meta property=\"og:description\" content=\"{_attr_escape(ogd)}\">"),
                }
            )

    return patches


# -----------------------------------------------------
# Implementatieplan — scoring & builders
# -----------------------------------------------------
def _sev_num(sev: str) -> int:
    s = (sev or "").lower()
    return 3 if s == "high" else (2 if s == "medium" else 1)


def _effort_label(n: int) -> str:
    return {1: "S", 2: "M", 3: "L"}.get(int(n or 1), "S")


def _impact_effort_for_text(field: str) -> Tuple[int, int]:
    f = (field or "").lower()
    if f == "title":
        return 4, 1
    if f == "meta_description":
        return 4, 1
    if f == "h1":
        return 2, 1
    if f == "intro_paragraph":
        return 3, 2
    return 2, 2


def _impact_effort_for_patch(category: str, problem: str) -> Tuple[int, int]:
    c = (category or "").lower()
    if c == "canonical":
        return 5, 1
    if c == "open_graph":
        return 2, 1
    return 2, 2


def _impact_effort_for_finding(finding: str) -> Tuple[int, int]:
    f = (finding or "").lower()
    if f.startswith("http error"):
        return 4, 3
    if f == "noindex set":
        return 5, 2
    if f == "orphan/low internal links":
        return 3, 2
    if f == "missing h1":
        return 3, 1
    return 2, 2


def _task_texts_for_text(field: str, lang: str) -> Tuple[str, str]:
    nl = lang == "nl"
    f = (field or "").lower()
    if f == "title":
        return ("Titel optimaliseren (uniek + lengte)" if nl else "Optimize title (unique + in-range)",
                "Verbetert CTR en duidelijkheid; voorkomt duplicaten." if nl else "Improves CTR and clarity; avoids duplicates.")
    if f == "meta_description":
        return ("Meta description toevoegen/optimaliseren" if nl else "Add/optimize meta description",
                "Betere snippet-kwaliteit en hogere doorklik." if nl else "Improves snippet quality and click-through.")
    if f == "h1":
        return ("H1 herschrijven (anders dan title)" if nl else "Rewrite H1 (not identical to title)",
                "Meer context voor gebruiker en zoekmachine." if nl else "Adds context for users and search engines.")
    if f == "intro_paragraph":
        return ("Intro-tekst toevoegen/verbeteren" if nl else "Add/improve intro paragraph",
                "≥250 woorden, antwoord-first; versterkt topical authority." if nl else "≥250 words, answer-first; strengthens topical authority.")
    return ("Tekstoptimalisatie", "Contentverbetering") if nl else ("Text optimisation", "Content improvement")


def _task_texts_for_patch(category: str, lang: str) -> Tuple[str, str]:
    nl = lang == "nl"
    c = (category or "").lower()
    if c == "canonical":
        return ("Canonical corrigeren", "Voorkomt duplicate/kanonieke inconsistentie; betere indexatie.") if nl else ("Fix canonical", "Avoids duplicate/canonical mismatch; improves indexing.")
    if c == "open_graph":
        return ("Open Graph instellen", "Nettere previews op social/AI; indirecte CTR.") if nl else ("Set up Open Graph", "Cleaner social/AI previews; indirect CTR benefits.")
    return ("Technische patch", "Verbetering in <head>") if nl else ("Technical patch", "Head improvement")


def _task_texts_for_finding(finding: str, lang: str) -> Tuple[str, str]:
    nl = lang == "nl"
    f = (finding or "").lower()
    if f.startswith("http error"):
        return ("HTTP-fout oplossen", "Niet-indexeerbaar en slechte UX; herstel endpoint of update links.") if nl else ("Fix HTTP error", "Not indexable and poor UX; fix endpoint or update links.")
    if f == "noindex set":
        return ("Noindex verwijderen", "Waardevolle pagina is uitgesloten; alleen gebruiken waar bedoeld.") if nl else ("Remove noindex", "Valuable page excluded; keep only where intended.")
    if f == "orphan/low internal links":
        return ("Interne links toevoegen", "Verhoogt crawlbaarheid en autoriteit; ≥2 inlinks.") if nl else ("Add internal links", "Improves crawlability and authority; ≥2 inlinks.")
    if f == "missing h1":
        return ("H1 toevoegen", "Duidelijk hoofdonderwerp voor bezoekers en bots.") if nl else ("Add H1", "Clear primary topic for users and bots.")
    return (("Technische SEO-fix: " + finding), "Los dit item op.") if nl else (("Technical SEO fix: " + finding), "Resolve this issue.")


def _plan_items(text_rows: List[Dict[str, Any]], tag_patches: List[Dict[str, Any]], technical_rows: List[Dict[str, Any]], pages: List[Dict[str, Any]], site_lang: Optional[str]) -> List[Dict[str, Any]]:
    by_url = {_norm_url(p.get("final_url") or p.get("url") or ""): p for p in (pages or [])}
    items: List[Dict[str, Any]] = []

    for r in text_rows:
        url = r["url"]
        lang = r.get("lang") or _detect_lang([(by_url.get(url) or {}).get("title") or "", (by_url.get(url) or {}).get("h1") or ""] + ((by_url.get(url) or {}).get("paragraphs") or []), site_lang)
        impact, effort = _impact_effort_for_text(r["field"])

        prob = (r.get("problem") or "").lower()
        if r["field"] == "title" and ("missing" in prob or "duplicate" in prob or "ontbreekt" in prob or "dubbel" in prob):
            severity = 3
        elif r["field"] == "title":
            severity = 2
        elif r["field"] == "meta_description" and ("missing" in prob or "ontbreekt" in prob):
            severity = 2
        elif r["field"] == "meta_description" and ("duplicate" in prob of "dubbel" in prob):
            severity = 2
        elif r["field"] == "h1":
            severity = 1
        elif r["field"] == "intro_paragraph":
            severity = 2
        else:
            severity = 1

        task, why = _task_texts_for_text(r["field"], lang)
        priority = round((impact + severity) / max(1, effort), 2)

        items.append(
            {
                "url": url,
                "task": task,
                "why": why,
                "field": r["field"],
                "category": "text",
                "severity": severity,
                "impact": impact,
                "effort": effort,
                "effort_label": _effort_label(effort),
                "priority": priority,
                "patchable": True,
                "html_patch": r.get("html_patch"),
                "lang": lang,
            }
        )

    for pch in tag_patches:
        url = pch["url"]
        page = by_url.get(url) or {}
        lang = _detect_lang([page.get("title") or "", page.get("h1") or "", page.get("meta_description") or ""] + (page.get("paragraphs") or []), site_lang)
        impact, effort = _impact_effort_for_patch(pch["category"], pch.get("problem") or "")
        sev = 2 if (pch["category"] == "canonical" and pch["problem"] == "differs from page URL") else (1 if pch["category"] in ("open_graph", "canonical") else 1)

        task, why = _task_texts_for_patch(pch["category"], lang)
        priority = round((impact + sev) / max(1, effort), 2)

        items.append(
            {
                "url": url,
                "task": task,
                "why": why,
                "field": pch["category"],
                "category": "tag",
                "severity": sev,
                "impact": impact,
                "effort": effort,
                "effort_label": _effort_label(effort),
                "priority": priority,
                "patchable": True,
                "html_patch": pch.get("html_patch"),
                "lang": lang,
            }
        )

    for r in technical_rows:
        if r["finding"] in _PATCHABLE_FINDINGS:
            continue
        url = r["url"]
        page = by_url.get(url) or {}
        lang = _detect_lang([page.get("title") or "", page.get("h1") or ""], site_lang)
        impact, effort = _impact_effort_for_finding(r["finding"])
        sev = _sev_num(r["severity"])

        task, why = _task_texts_for_finding(r["finding"], lang)
        priority = round((impact + sev) / max(1, effort), 2)

        items.append(
            {
                "url": url,
                "task": task,
                "why": why,
                "field": r["finding"],
                "category": "technical",
                "severity": sev,
                "impact": impact,
                "effort": effort,
                "effort_label": _effort_label(effort),
                "priority": priority,
                "patchable": False,
                "html_patch": None,
                "lang": lang,
            }
        )

    items.sort(key=lambda x: (-x["priority"], x["effort"], x["url"]))
    return items


# -----------------------------------------------------
# CSV/JSON export helpers
# -----------------------------------------------------
def _csv_base64(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    buff = io.StringIO()
    writer = csv.writer(buff)
    writer.writerow(columns)
    for r in rows:
        writer.writerow([r.get(c, "") if not isinstance(r.get(c, ""), (list, dict)) else json.dumps(r.get(c, "")) for c in columns])
    return base64.b64encode(buff.getvalue().encode("utf-8")).decode("utf-8")


def _patch_rows_for_export(text_rows: List[Dict[str, Any]], tag_patches: List[Dict[str, Any]], site_lang: Optional[str], extra_patches: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for r in text_rows:
        impact, effort = _impact_effort_for_text(r["field"])
        prob = (r.get("problem") or "").lower()
        if r["field"] == "title" and ("missing" in prob or "duplicate" in prob or "ontbreekt" in prob or "dubbel" in prob):
            severity = 3
        elif r["field"] == "title":
            severity = 2
        elif r["field"] == "meta_description" and ("missing" in prob or "ontbreekt" in prob):
            severity = 2
        elif r["field"] == "meta_description" and ("duplicate" in prob or "dubbel" in prob):
            severity = 2
        elif r["field"] == "h1":
            severity = 1
        elif r["field"] == "intro_paragraph":
            severity = 2
        else:
            severity = 1
        priority = round((impact + severity) / max(1, effort), 2)

        out.append(
            {
                "url": r["url"],
                "field": r["field"],
                "problem": r.get("problem"),
                "current": r.get("current"),
                "proposed": r.get("proposed"),
                "html_patch": r.get("html_patch"),
                "category": "head" if r["field"] in ("title", "meta_description") else "body",
                "severity": {3: "high", 2: "medium", 1: "low"}[severity],
                "impact": impact,
                "effort": effort,
                "priority": priority,
                "patchable": True,
            }
        )

    for pch in tag_patches:
        impact, effort = _impact_effort_for_patch(pch["category"], pch.get("problem") or "")
        sev = 2 if (pch["category"] == "canonical" and pch.get("problem") == "differs from page URL") else (1 if pch["category"] in ("open_graph", "canonical") else 1)
        priority = round((impact + sev) / max(1, effort), 2)
        out.append(
            {
                "url": pch["url"],
                "field": pch["category"],
                "problem": pch.get("problem"),
                "current": pch.get("current"),
                "proposed": None,
                "html_patch": pch.get("html_patch"),
                "category": "head",
                "severity": {3: "high", 2: "medium", 1: "low"}[sev],
                "impact": impact,
                "effort": effort,
                "priority": priority,
                "patchable": True,
            }
        )

    for ap in (extra_patches or []):
        sev_num = 2
        sev_lbl = (ap.get("severity") or "medium")
        if isinstance(ap.get("severity"), int):
            sev_num = ap["severity"]
            sev_lbl = {3: "high", 2: "medium", 1: "low"}.get(sev_num, "medium")
        out.append(
            {
                "url": ap.get("url"),
                "field": ap.get("field"),
                "problem": ap.get("problem"),
                "current": ap.get("current"),
                "proposed": ap.get("proposed"),
                "html_patch": ap.get("html_patch"),
                "category": ap.get("category") or "body",
                "severity": sev_lbl,
                "impact": ap.get("impact", 3),
                "effort": ap.get("effort", 2),
                "priority": ap.get("priority", 5.0),
                "patchable": ap.get("patchable", True),
            }
        )

    return out


# -----------------------------------------------------
# Optional LLM executive summary
# -----------------------------------------------------
def _try_llm_summary(site_meta, seo_rows, geo_rows, aeo_rows, aeo_scorecards=None) -> Optional[str]:
    if not _openai_client:
        return None
    lang = (site_meta.get("language") or "en").lower()
    is_nl = lang.startswith("nl")
    try:
        if is_nl:
            sys = "Je bent een senior SEO/GEO/AEO-auditor. Schrijf 2–4 korte alinea’s in het Nederlands. Wees concreet, kwantificeer waar mogelijk, en prioriteer fixes."
        else:
            sys = "You are a senior SEO/GEO/AEO auditor. Write 2–4 short paragraphs in English. Be concrete, quantify where possible, and prioritize fixes."
        payload = {
            "site": {
                "name": site_meta.get("account_name"),
                "url": site_meta.get("url"),
                "language": site_meta.get("language"),
                "country": site_meta.get("country"),
            },
            "top": {
                "seo": [r["finding"] for r in seo_rows[:3]],
                "geo": [r["finding"] for r in geo_rows[:3]],
                "aeo": [r["finding"] for r in aeo_rows[:3]],
            },
            "aeo_scores": (aeo_scorecards or [])[:5],
            "counts": {"seo": len(seo_rows), "geo": len(geo_rows), "aeo": len(aeo_rows)},
        }
        user = json.dumps(payload, ensure_ascii=False)
        resp = _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            timeout=OPENAI_TIMEOUT_SEC,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


# -----------------------------------------------------
# Report generation
# -----------------------------------------------------
def generate_report(conn, job):
    site_id = job["site_id"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    site_meta = _fetch_site_meta(conn, site_id)
    crawl = _fetch_latest_job(conn, site_id, "crawl")
    keywords = _fetch_latest_job(conn, site_id, "keywords")
    faq = _fetch_latest_job(conn, site_id, "faq")
    schema_job = _fetch_latest_job(conn, site_id, "schema")
    aeo_job = _fetch_latest_job(conn, site_id, "aeo")

    seo_rows_all, unique_pages = _build_seo_findings(crawl)
    text_rows = _build_text_findings(crawl, site_meta.get("language"))
    dup_groups = _build_duplicate_groups(crawl, site_meta.get("language"))
    technical_rows = [r for r in seo_rows_all if r["finding"] not in _TEXT_FINDINGS_ALL and r["finding"] not in _PATCHABLE_FINDINGS]
    geo_rows = _geo_recommendations(site_meta, crawl, schema_job)
    tag_patches = _build_canonical_og_patches(crawl, text_rows)

    aeo_concrete = _aeo_from_job(aeo_job)
    aeo_scorecards = aeo_concrete.get("scorecards", [])
    aeo_qna_rows = aeo_concrete.get("rows", [])
    aeo_patches = aeo_concrete.get("patches", [])
    aeo_plan_items = aeo_concrete.get("plan_items", [])

    aeo_quality_rows = _aeo_findings_from_faq(faq, crawl)
    aeo_quality_rows += _aeo_findings_from_aeo_job(aeo_concrete)

    plan_base = _plan_items(text_rows, tag_patches, technical_rows, unique_pages, site_meta.get("language"))
    plan_items = plan_base + aeo_plan_items
    plan_items.sort(key=lambda x: (-x["priority"], x.get("effort", 1), x["url"]))

    patches_export_rows = _patch_rows_for_export(text_rows, tag_patches, site_meta.get("language"), extra_patches=aeo_patches)
    patches_csv_b64 = _csv_base64(
        patches_export_rows,
        ["url", "field", "problem", "current", "proposed", "html_patch", "category", "severity", "impact", "effort", "priority", "patchable"],
    )
    plan_csv_b64 = _csv_base64(
        plan_items,
        ["url", "task", "why", "category", "field", "severity", "impact", "effort", "effort_label", "priority", "patchable", "html_patch", "lang"],
    )

    exec_summary = _try_llm_summary(site_meta, seo_rows_all, geo_rows, aeo_quality_rows, aeo_scorecards) or (
        "Dit rapport bevat concrete issues en aanbevelingen voor SEO (techniek), GEO (entity/schema) en AEO (answer readiness), inclusief acceptatiecriteria."
        if (site_meta.get("language") or "").lower().startswith("nl")
        else "This report lists concrete issues and recommendations across SEO, GEO and AEO with acceptance criteria."
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    width = doc.width
    elems: List[Any] = []
    S = _styles()

    title = f"SEO • GEO • AEO Audit — {site_meta.get('url') or ''}"
    elems.append(Paragraph(title, S["Title"]))
    elems.append(Paragraph(f"Generated: {now}", S["Normal"]))
    elems.append(Spacer(1, 8))
    elems.append(Paragraph(exec_summary, S["Small"]))
    elems.append(PageBreak())

    # AEO scorecards
    elems.append(Paragraph("AEO — Answer readiness (scorecards)", S["Heading2"]))
    if not aeo_scorecards:
        elems.append(Paragraph("No AEO job output yet.", S["Normal"]))
    else:
        headers = ["Page URL", "Score (0–100)", "Issues", "Metrics"]
        colw = [0.32 * width, 0.14 * width, 0.28 * width, 0.26 * width]
        data = [headers]
        for r in aeo_scorecards:
            metrics_txt = ", ".join([f"{k}:{v}" for k, v in (r.get("metrics") or {}).items()])
            issues_txt = ", ".join(r.get("issues") or []) or "OK"
            url_txt = r.get("url") or ""
            if r.get("type"):
                url_txt = f"{url_txt}  [{r.get('type')}]"
            data.append([P(_shorten(url_txt, 120)), P(str(r.get("score", "")), "Tiny"), P(issues_txt, "Tiny"), P(metrics_txt, "Tiny")])
        elems.append(_make_table(data, colw))
    elems.append(Spacer(1, 8))

    # AEO Q&A
    elems.append(Paragraph("AEO — Q&A (snippet-ready, ready-to-paste)", S["Heading2"]))
    if not aeo_qna_rows:
        elems.append(Paragraph("No Q&A candidates generated (FAQ pages only).", S["Normal"]))
    else:
        headers = ["Page URL", "Q", "Proposed answer (≤80 words)", "Gaps"]
        colw = [0.28 * width, 0.26 * width, 0.30 * width, 0.16 * width]
        data = [headers]
        for r in aeo_qna_rows[:80]:
            data.append([P(_shorten(r["url"], 120)), P(r["q"]), P(r["a"]), P(r.get("gaps") or "OK", "Tiny")])
        elems.append(_make_table(data, colw))
    elems.append(PageBreak())

    # SEO text fixes
    elems.append(Paragraph("SEO — Concrete text fixes (ready-to-paste)", S["Heading2"]))
    if not text_rows:
        elems.append(Paragraph("No text issues found in titles/meta/H1/intro." if (site_meta.get("language") or "").lower().startswith("en") else "Geen tekstuele problemen gevonden in titels/meta/H1/intro.", S["Normal"]))
    else:
        headers = ["Page URL", "Field", "Issue", "Current", "Proposed (ready-to-paste)"] if (site_meta.get("language") or "").lower().startswith("en") else ["Page URL", "Veld", "Probleem", "Huidig", "Voorstel (ready-to-paste)"]
        colw = [0.23 * width, 0.10 * width, 0.15 * width, 0.24 * width, 0.28 * width]
        data = [headers]
        for r in text_rows:
            data.append(
                [
                    P(_shorten(r["url"], 120)),
                    P(r["field"]),
                    P(r["problem"], "Tiny"),
                    P(_shorten(str(r.get("current", "")), 300), "Tiny"),
                    P(_shorten(str(r.get("proposed", "")), 300), "Tiny"),
                ]
            )
        elems.append(_make_table(data, colw))
        elems.append(Spacer(1, 6))
        elems.append(Paragraph("HTML patches (copy & paste):" if (site_meta.get("language") or "").lower().startswith("en") else "HTML-patches (kopieer & plak):", S["Small"]))
        for r in text_rows[:12]:
            elems.append(KeepTogether([P(f"{r['field']} — {r['url']}", "Tiny"), Code(r["html_patch"]), Spacer(1, 4)]))
    elems.append(PageBreak())

    # Duplicate groups
    elems.append(Paragraph("Duplicate groups — titles", S["Heading2"]))
    if not dup_groups["title_groups"]:
        elems.append(Paragraph("No duplicate titles found." if (site_meta.get("language") or "").lower().startswith("en") else "Geen dubbele titles gevonden.", S["Normal"]))
    else:
        for g in dup_groups["title_groups"]:
            elems.append(Paragraph(f"Current title (×{g['count']}): {xml_escape(_shorten(g['value'], 120))}" if (site_meta.get("language") or "").lower().startswith("en") else f"Huidige title (×{g['count']}): {xml_escape(_shorten(g['value'], 120))}", S["Small"]))
            headers = ["Page URL", "Unique proposal"] if (site_meta.get("language") or "").lower().startswith("en") else ["Page URL", "Uniek voorstel"]
            colw = [0.55 * width, 0.45 * width]
            data = [headers]
            for row in g["rows"]:
                data.append([P(_shorten(row["url"], 90)), P(row["proposed"], "Tiny")])
            elems.append(_make_table(data, colw))
            elems.append(Spacer(1, 6))
    elems.append(PageBreak())

    elems.append(Paragraph("Duplicate groups — meta descriptions", S["Heading2"]))
    if not dup_groups["meta_groups"]:
        elems.append(Paragraph("No duplicate meta descriptions found." if (site_meta.get("language") or "").lower().startswith("en") else "Geen dubbele meta descriptions gevonden.", S["Normal"]))
    else:
        for g in dup_groups["meta_groups"]:
            elems.append(Paragraph(f"Current meta (×{g['count']}): {xml_escape(_shorten(g['value'], 180))}" if (site_meta.get("language") or "").lower().startswith("en") else f"Huidige meta (×{g['count']}): {xml_escape(_shorten(g['value'], 180))}", S["Small"]))
            headers = ["Page URL", "Unique proposal"] if (site_meta.get("language") or "").lower().startswith("en") else ["Page URL", "Uniek voorstel"]
            colw = [0.55 * width, 0.45 * width]
            data = [headers]
            for row in g["rows"]:
                data.append([P(_shorten(row["url"], 90)), P(row["proposed"], "Tiny")])
            elems.append(_make_table(data, colw))
            elems.append(Spacer(1, 6))
    elems.append(PageBreak())

    # Canonical & Open Graph patches — tabel ZONDER patch-code
    elems.append(Paragraph("HTML patches — Canonical & Open Graph", S["Heading2"]))
    if not tag_patches:
        elems.append(Paragraph("No patches needed." if (site_meta.get("language") or "").lower().startswith("en") else "Geen patches nodig.", S["Normal"]))
    else:
        headers = ["Page URL", "Category", "Issue", "Current"] if (site_meta.get("language") or "").lower().startswith("en") else ["Page URL", "Categorie", "Probleem", "Huidig"]
        colw = [0.35 * width, 0.15 * width, 0.20 * width, 0.30 * width]
        data = [headers]
        for pch in tag_patches:
            data.append([P(_shorten(pch["url"], 120)), P(pch["category"]), P(pch["problem"], "Tiny"), P(_shorten(pch["current"], 250), "Tiny")])
        elems.append(_make_table(data, colw))
        elems.append(Spacer(1, 6))
        # Patch-codeblokken los onder de tabel (voorkomt paraparser issues)
        elems.append(Paragraph("Patches (copy & paste):" if (site_meta.get("language") or "").lower().startswith("en") else "Patches (kopieer & plak):", S["Small"]))
        for pch in tag_patches[:40]:
            label = f"{pch['category']} — {pch['url']}"
            elems.append(KeepTogether([P(_shorten(label, 140), "Tiny"), Code(pch.get("html_patch") or ""), Spacer(1, 4)]))
    elems.append(PageBreak())

    # Implementation plan
    impl_title = "Implementation plan (priority × impact × effort)" if (site_meta.get("language") or "").lower().startswith("en") else "Implementatieplan (prioriteit × impact × effort)"
    elems.append(Paragraph(impl_title, S["Heading2"]))
    if not plan_items:
        elems.append(Paragraph("No tasks generated." if (site_meta.get("language") or "").lower().startswith("en") else "Geen taken gegenereerd.", S["Normal"]))
    else:
        headers = ["Status", "URL", "Task", "Why/impact", "Effort", "Priority", "Patch"] if (site_meta.get("language") or "").lower().startswith("en") else ["Status", "URL", "Taak", "Waarom/impact", "Effort", "Prioriteit", "Patch"]
        colw = [0.07 * width, 0.23 * width, 0.20 * width, 0.25 * width, 0.09 * width, 0.08 * width, 0.08 * width]
        data = [headers]
        for it in plan_items[:60]:
            checkbox = "☐"
            patch_note = "yes" if it.get("patchable") and it.get("html_patch") else "—"
            if not (site_meta.get("language") or "").lower().startswith("en"):
                patch_note = "ja" if it.get("patchable") and it.get("html_patch") else "—"
            data.append([P(checkbox, "Tiny"), P(_shorten(it["url"], 110), "Tiny"), P(_shorten(it["task"], 120), "Tiny"), P(_shorten(it["why"], 180), "Tiny"), P(it["effort_label"], "Tiny"), P(str(it["priority"]), "Tiny"), P(patch_note, "Tiny")])
        elems.append(_make_table(data, colw))
    elems.append(PageBreak())

    # Other technical issues
    elems.append(Paragraph("SEO — Other technical issues", S["Heading2"]))
    if not technical_rows:
        elems.append(Paragraph("No other technical issues." if (site_meta.get("language") or "").lower().startswith("en") else "Geen overige technische issues.", S["Normal"]))
    else:
        headers = ["Page URL", "Finding", "Severity", "Fix (summary)", "Acceptance Criteria"]
        colw = [0.26 * width, 0.24 * width, 0.12 * width, 0.18 * width, 0.20 * width]
        data = [headers]
        for r in technical_rows:
            data.append([P(_shorten(r["url"], 120)), P(r["finding"]), P(r["severity"].title(), "Tiny"), P(r["fix"]), P("• " + "<br/>• ".join([xml_escape(x) for x in r["accept"]]), "Tiny")])
        elems.append(_make_table(data, colw))
    elems.append(PageBreak())

    # GEO Recommendations
    elems.append(Paragraph("GEO Recommendations (entity/schema)", S["Heading2"]))
    if not geo_rows:
        elems.append(Paragraph("No schema/entity recommendations." if (site_meta.get("language") or "").lower().startswith("en") else "Geen schema/entity-aanbevelingen.", S["Normal"]))
    else:
        headers = ["Page URL", "Recommendation", "Severity", "Fix (summary)", "Acceptance Criteria"]
        colw = [0.26 * width, 0.24 * width, 0.12 * width, 0.18 * width, 0.20 * width]
        data = [headers]
        for r in geo_rows:
            data.append([P(_shorten(r["url"], 120)), P(r["finding"]), P(r["severity"].title(), "Tiny"), P(r["fix"]), P("• " + "<br/>• ".join([xml_escape(x) for x in r["accept"]]), "Tiny")])
        elems.append(_make_table(data, colw))
    elems.append(PageBreak())

    # AEO Findings
    elems.append(Paragraph("AEO Findings (answer & citation readiness)", S["Heading2"]))
    if not aeo_quality_rows:
        elems.append(Paragraph("No AEO findings." if (site_meta.get("language") or "").lower().startswith("en") else "Geen AEO-issues.", S["Normal"]))
    else:
        headers = ["Page URL", "Finding", "Severity", "Fix (summary)", "Acceptance Criteria"]
        colw = [0.26 * width, 0.24 * width, 0.12 * width, 0.18 * width, 0.20 * width]
        data = [headers]
        for r in aeo_quality_rows:
            data.append([P(_shorten(r["url"], 120)), P(r["finding"]), P(r["severity"].title(), "Tiny"), P(r["fix"]), P("• " + "<br/>• ".join([xml_escape(x) for x in r["accept"]]), "Tiny")])
        elems.append(_make_table(data, colw))
    elems.append(PageBreak())

    # Appendix — JSON-LD snippets
    elems.append(Paragraph("Appendix — JSON-LD Snippets (paste & adapt)", S["Heading2"]))
    brand = site_meta.get("account_name") or "YourBrand"
    home = (site_meta.get("url") or "").strip().rstrip("/") or "https://example.com"
    snippets = [
        ("Organization + WebSite", {"@context": "https://schema.org", "@type": "Organization", "name": brand, "url": f"{home}/", "logo": f"{home}/favicon-512.png", "sameAs": ["https://www.linkedin.com/company/yourbrand", "https://www.wikidata.org/wiki/QXXXXX"]}),
        ("WebSite", {"@context": "https://schema.org", "@type": "WebSite", "name": brand, "url": f"{home}/"}),
        ("BreadcrumbList (example)", {"@context": "https://schema.org", "@type": "BreadcrumbList", "itemListElement": [{"@type": "ListItem", "position": 1, "name": "Home", "item": f"{home}/"}]}),
    ]
    for title_txt, obj in snippets:
        elems.append(Paragraph(title_txt, S["H3tight"]))
        elems.append(KeepTogether([Code(json.dumps(obj, indent=2, ensure_ascii=False)), Spacer(1, 6)]))

    if aeo_job:
        elems.append(Paragraph("Appendix — AEO JSON-LD per page", S["Heading2"]))
        for page in (aeo_job.get("pages") or []):
            pretty = json.dumps(page.get("faq_jsonld") or {}, indent=2, ensure_ascii=False)
            if pretty.strip() and pretty != "{}":
                elems.append(Paragraph(_shorten(_norm_url(page.get("url") or ""), 120), S["H3tight"]))
                elems.append(KeepTogether([Code(pretty[:4000]), Spacer(1, 6)]))

    if schema_job and schema_job.get("schema"):
        elems.append(Paragraph("Generated (from jobs.schema)", S["H3tight"]))
        pretty = json.dumps(schema_job["schema"], indent=2, ensure_ascii=False)
        elems.append(KeepTogether([Code(pretty[:4000]), Spacer(1, 6)]))

    doc.build(elems)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
    return {
        "pdf_base64": pdf_base64,
        "meta": {
            "site_id": str(site_id),
            "generated_at": now,
            "sections": {
                "aeo_scorecards": bool(aeo_scorecards),
                "aeo_qas": bool(aeo_qna_rows),
                "text_fixes": bool(text_rows),
                "duplicate_titles": bool(dup_groups["title_groups"]),
                "duplicate_metas": bool(dup_groups["meta_groups"]),
                "tag_patches": bool(tag_patches),
                "implementation_plan": bool(plan_items),
                "seo_other": bool(technical_rows),
                "geo_recommendations": bool(geo_rows),
                "aeo_findings": bool(aeo_quality_rows),
            },
        },
        "patches_json": _patch_rows_for_export(text_rows, tag_patches, site_meta.get("language"), extra_patches=aeo_patches),
        "patches_csv_base64": patches_csv_b64,
        "plan_items": plan_items,
        "plan_csv_base64": plan_csv_b64,
    }
