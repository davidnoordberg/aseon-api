# report_agent.py
import os
import io
import re
import json
import base64
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

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

# Optional LLM synthesis
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
    if not u: return ""
    p = urlsplit(u)
    scheme = p.scheme or "https"
    host = p.hostname or ""
    path = (p.path or "/")
    if not path: path = "/"
    if not path.startswith("/"): path = "/" + path
    if path != "/" and path.endswith("/"): path = path[:-1]
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

# -----------------------------------------------------
# Findings builders
# -----------------------------------------------------
def _build_link_graph(pages: List[Dict[str, Any]]) -> Dict[str, int]:
    inlinks = {}
    url_index = {}
    for p in pages:
        u = _norm_url(p.get("final_url") or p.get("url") or "")
        url_index[u] = True
    for p in pages:
        src = _norm_url(p.get("final_url") or p.get("url") or "")
        for l in (p.get("links") or []):
            tgt = _norm_url(l)
            if tgt in url_index:
                inlinks[tgt] = inlinks.get(tgt, 0) + 1
    return inlinks

def _dup_map(values: List[Optional[str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for v in values:
        if not v: continue
        k = v.strip().lower()
        if not k: continue
        counts[k] = counts.get(k, 0) + 1
    return counts

def _seo_findings_from_crawl(crawl: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not crawl:
        return []
    pages = crawl.get("pages") or []
    inlinks = _build_link_graph(pages)
    title_counts = _dup_map([p.get("title") for p in pages if p.get("status") == 200])
    meta_counts  = _dup_map([p.get("meta_description") for p in pages if p.get("status") == 200])

    out: List[Dict[str, Any]] = []
    for p in pages:
        url = p.get("final_url") or p.get("url") or ""
        u = _norm_url(url)
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

        # HTTP status
        if status >= 400:
            add("HTTP error " + str(status), "high",
                "Fix the endpoint to return 200 or remove from internal linking.",
                ["URL returns 200", "Internal links updated"])

        # Title
        if not title:
            add("Missing <title>", "high",
                "Add a descriptive, unique <title> (30–65 chars) that matches page intent.",
                ["<title> present", "30–65 chars", "Unique site-wide"])
        else:
            if len(title) < 30 or len(title) > 65:
                add("Title length suboptimal", "medium",
                    "Keep <title> between ~30–65 chars; lead with primary intent/entity.",
                    ["30–65 chars", "Primary intent first"])
            if title_counts.get(title.strip().lower(), 0) > 1:
                add("Duplicate <title>", "high",
                    "Make the title unique per page; differentiate with intent or entities.",
                    ["Titles unique site-wide"])

        # Meta description
        if not md:
            add("Missing meta description", "medium",
                "Add unique <meta name='description'> (~140–160 chars) that summarizes the answer/value.",
                ["Tag present", "140–160 chars", "Unique on site"])
        else:
            if len(md) < 70 or len(md) > 180:
                add("Meta description length suboptimal", "low",
                    "Aim for ~140–160 chars; avoid truncation and thin summaries.",
                    ["~140–160 chars"])
            if meta_counts.get(md.strip().lower(), 0) > 1:
                add("Duplicate meta description", "medium",
                    "Rewrite to reflect page’s unique value; avoid reuse.",
                    ["Descriptions unique site-wide"])

        # H1
        if not h1:
            add("Missing H1", "high",
                "Add exactly one <h1> with the primary topic; keep 30–65 chars.",
                ["Exactly one <h1>", "Matches intent"])
        elif title and h1.strip().lower() == title.strip().lower():
            add("H1 duplicates title", "low",
                "Differentiate H1 slightly to add context; keep user-first phrasing.",
                ["H1 present", "Not identical to title"])

        # Robots
        if noindex:
            add("noindex set", "high",
                "Remove noindex for indexable pages; keep only on intentional pages (thank-you, internal tools).",
                ["No 'noindex' on valuable pages"])

        # Canonical
        if not canon:
            add("Missing canonical", "low",
                "Add <link rel='canonical'> to the preferred URL to prevent duplicates.",
                ["Canonical present", "Equals preferred URL"])
        else:
            if _norm_url(canon) != u:
                add("Canonical differs from page URL", "medium",
                    "Canonical should match the preferred URL; align host & path.",
                    ["Canonical equals preferred URL"])

        # Word count
        if word_count and word_count < 250:
            add("Low visible text", "medium",
                "Add concise, answer-first copy and supporting evidence; target ≥250–400 words for key pages.",
                ["≥250 words visible", "Lead with answer"])

        # Open Graph
        if not og_title or not og_desc:
            add("Missing Open Graph tags", "low",
                "Add og:title and og:description for clean sharing/AI snippets.",
                ["og:title present", "og:description present"])

        # Internal links
        if inl == 0:
            add("Orphan/low internal links", "medium",
                "Link to this page from at least 2 relevant pages using natural anchor text.",
                ["≥2 internal inlinks", "Anchors descriptive"])

    return out

def _aeo_findings_from_faq(faq: Optional[Dict[str, Any]], crawl: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    # FAQ job quality
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
                        "accept": [
                            "Each FAQ has a working source URL",
                            "Source resolves with HTTP 200",
                        ],
                    }
                )
            if len(a.split()) > 90:
                out.append(
                    {
                        "url": (faq.get("site") or {}).get("url") or "",
                        "finding": f"FAQ answer too long — “{_shorten(q, 60)}”",
                        "severity": "low",
                        "fix": "Trim to ≤80 words; lead with the answer; keep nouns/verbs concrete.",
                        "accept": [
                            "≤80 words",
                            "Lead sentence answers the question",
                        ],
                    }
                )
    # Site FAQ schema presence (from crawl JSON-LD types)
    has_faq = False
    if crawl:
        for p in (crawl.get("pages") or []):
            types = [t.lower() for t in (p.get("jsonld_types") or [])]
            if any(t == "faqpage" for t in types):
                has_faq = True
                break
    if not has_faq:
        out.append({
            "url": (crawl or {}).get("start_url") or "",
            "finding": "No FAQPage schema detected",
            "severity": "medium",
            "fix": "Add compact FAQ blocks (3–6 Q/As) on relevant pages with FAQPage JSON-LD.",
            "accept": ["Rich Results Test: valid FAQPage", "Answers ≤80 words", "Each item has source link"]
        })
    return out

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
                "accept": [
                    "Rich Results Test: valid Organization & WebSite",
                    "Logo URL HTTPS and public",
                    "≥2 sameAs profiles resolve"
                ],
            }
        )
    if "breadcrumblist" not in present_types:
        recs.append(
            {
                "url": home.rstrip("/") + "/faq",
                "finding": "Add BreadcrumbList on sections/FAQ",
                "severity": "medium",
                "fix": "Provide BreadcrumbList JSON-LD reflecting on-page breadcrumbs.",
                "accept": [
                    "Valid BreadcrumbList",
                    "Positions sequential (1..n)",
                    "URLs canonical"
                ],
            }
        )
    if schema_job and schema_job.get("biz_type","").lower() == "faqpage":
        recs.append(
            {
                "url": home,
                "finding": "Broaden schema coverage beyond FAQPage",
                "severity": "medium",
                "fix": "Add Article/BlogPosting where applicable; add Person for authors; link entities with about/mentions.",
                "accept": [
                    "Article/BlogPosting valid on content pages",
                    "Author (Person) present with sameAs where appropriate"
                ],
            }
        )
    return recs

# -----------------------------------------------------
# Optional LLM executive summary
# -----------------------------------------------------
def _try_llm_summary(site_meta, seo_rows, geo_rows, aeo_rows) -> Optional[str]:
    if not _openai_client:
        return None
    try:
        sys = (
            "You are a senior SEO/GEO/AEO auditor. Write 2–4 short paragraphs. "
            "Be concrete, quantify where possible, and prioritise fixes."
        )
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
            "counts": {
                "seo": len(seo_rows),
                "geo": len(geo_rows),
                "aeo": len(aeo_rows),
            }
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

    # Inputs
    site_meta = _fetch_site_meta(conn, site_id)
    crawl = _fetch_latest_job(conn, site_id, "crawl")
    keywords = _fetch_latest_job(conn, site_id, "keywords")
    faq = _fetch_latest_job(conn, site_id, "faq")
    schema_job = _fetch_latest_job(conn, site_id, "schema")

    # Findings
    seo_rows = _seo_findings_from_crawl(crawl)
    aeo_rows = _aeo_findings_from_faq(faq, crawl)
    geo_rows = _geo_recommendations(site_meta, crawl, schema_job)

    # Executive summary
    exec_summary = _try_llm_summary(site_meta, seo_rows, geo_rows, aeo_rows) or (
        "This report lists concrete issues and recommendations across SEO (technical), "
        "GEO (entity/schema), and AEO (answer readiness). Each item includes acceptance criteria "
        "so your team can implement and verify fixes."
    )

    # --------- PDF ----------
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

    # Cover
    title = f"SEO • GEO • AEO Audit — {site_meta.get('url') or ''}"
    elems.append(Paragraph(title, S["Title"]))
    elems.append(Paragraph(f"Generated: {now}", S["Normal"]))
    elems.append(Spacer(1, 8))
    elems.append(Paragraph(exec_summary, S["Small"]))
    elems.append(PageBreak())

    # --- SEO Findings ---
    elems.append(Paragraph("SEO Findings (per page)", S["Heading2"]))
    if not seo_rows:
        elems.append(Paragraph("No findings from crawl.", S["Normal"]))
    else:
        headers = ["Page URL", "Finding", "Severity", "Fix (summary)", "Acceptance Criteria"]
        colw = [0.26 * width, 0.24 * width, 0.10 * width, 0.20 * width, 0.20 * width]
        data = [headers]
        for r in seo_rows:
            data.append(
                [
                    P(_shorten(r["url"], 120)),
                    P(r["finding"]),
                    P(r["severity"].title(), "Tiny"),
                    P(r["fix"]),
                    P("• " + "<br/>• ".join([xml_escape(x) for x in r["accept"]]), "Tiny"),
                ]
            )
        elems.append(_make_table(data, colw))
    elems.append(PageBreak())

    # --- GEO Recommendations (entity/schema) ---
    elems.append(Paragraph("GEO Recommendations (entity/schema)", S["Heading2"]))
    if not geo_rows:
        elems.append(Paragraph("No schema/entity recommendations.", S["Normal"]))
    else:
        headers = ["Page URL", "Recommendation", "Severity", "Fix (summary)", "Acceptance Criteria"]
        colw = [0.26 * width, 0.24 * width, 0.10 * width, 0.20 * width, 0.20 * width]
        data = [headers]
        for r in geo_rows:
            data.append(
                [
                    P(_shorten(r["url"], 120)),
                    P(r["finding"]),
                    P(r["severity"].title(), "Tiny"),
                    P(r["fix"]),
                    P("• " + "<br/>• ".join([xml_escape(x) for x in r["accept"]]), "Tiny"),
                ]
            )
        elems.append(_make_table(data, colw))
    elems.append(PageBreak())

    # --- AEO Findings ---
    elems.append(Paragraph("AEO Findings (answer & citation readiness)", S["Heading2"]))
    if not aeo_rows:
        elems.append(Paragraph("No AEO findings.", S["Normal"]))
    else:
        headers = ["Page URL", "Finding", "Severity", "Fix (summary)", "Acceptance Criteria"]
        colw = [0.26 * width, 0.24 * width, 0.10 * width, 0.20 * width, 0.20 * width]
        data = [headers]
        for r in aeo_rows:
            data.append(
                [
                    P(_shorten(r["url"], 120)),
                    P(r["finding"]),
                    P(r["severity"].title(), "Tiny"),
                    P(r["fix"]),
                    P("• " + "<br/>• ".join([xml_escape(x) for x in r["accept"]]), "Tiny"),
                ]
            )
        elems.append(_make_table(data, colw))
    elems.append(PageBreak())

    # --- Appendix: JSON-LD Snippets (paste & adapt) ---
    elems.append(Paragraph("Appendix — JSON-LD Snippets (paste & adapt)", S["Heading2"]))
    brand = site_meta.get("account_name") or "YourBrand"
    home = (site_meta.get("url") or "").strip().rstrip("/") or "https://example.com"
    snippets = [
        ("Organization + WebSite",
         {
           "@context":"https://schema.org","@type":"Organization",
           "name": brand, "url": f"{home}/", "logo": f"{home}/favicon-512.png",
           "sameAs": ["https://www.linkedin.com/company/yourbrand","https://www.wikidata.org/wiki/QXXXXX"]
         }),
        ("WebSite",
         {"@context":"https://schema.org","@type":"WebSite","name":brand,"url":f"{home}/"}),
        ("BreadcrumbList (example)",
         {"@context":"https://schema.org","@type":"BreadcrumbList",
          "itemListElement":[{"@type":"ListItem","position":1,"name":"Home","item":f"{home}/"}]})
    ]
    for title_txt, obj in snippets:
        elems.append(Paragraph(title_txt, S["H3tight"]))
        elems.append(KeepTogether([Code(json.dumps(obj, indent=2, ensure_ascii=False)), Spacer(1, 6)]))

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
                "seo_findings": bool(seo_rows),
                "geo_recommendations": bool(geo_rows),
                "aeo_findings": bool(aeo_rows),
            },
        },
    }
