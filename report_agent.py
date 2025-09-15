# report_agent.py
import os
import io
import json
import base64
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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
    KeepInFrame,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from xml.sax.saxutils import escape as xml_escape

# Optional LLM synthesis (safe fallback if it fails)
from openai import OpenAI

OPENAI_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "30"))
_openai_key = os.getenv("OPENAI_API_KEY")
_openai_client = OpenAI(api_key=_openai_key) if _openai_key else None


# ---------------------------
# DB helpers
# ---------------------------
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


# ---------------------------
# PDF helpers
# ---------------------------
def _styles():
    styles = getSampleStyleSheet()
    # Custom styles (unique names, no collisions). Use wordWrap='CJK' for better wrapping.
    if "Small" not in styles.byName:
        styles.add(ParagraphStyle(name="Small", fontSize=9, leading=12, wordWrap="CJK"))
    if "Tiny" not in styles.byName:
        styles.add(ParagraphStyle(name="Tiny", fontSize=8, leading=10, wordWrap="CJK"))
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
                ("ALIGN", (-1, 1), (-1, -1), "LEFT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("WORDWRAP", (0, 0), (-1, -1), "CJK"),
            ]
        )
    )
    return t


def _shorten(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    return (s[: max_chars - 1] + "…") if len(s) > max_chars else s


# ---------------------------
# Findings builders
# ---------------------------
def _seo_findings_from_crawl(crawl: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not crawl:
        return []
    ISSUE_MAP = {
        "missing_meta_description": {
            "finding": "Missing meta description",
            "severity": "medium",
            "fix": "Add a unique <meta name='description'> (140–160 chars) matching visible content; avoid duplicates.",
            "accept": [
                "Tag present in <head>",
                "Unique across the site",
                "Summarizes primary intent",
            ],
        },
        "missing_h1": {
            "finding": "Missing H1",
            "severity": "high",
            "fix": "Add exactly one <h1> that states the page’s primary topic; 30–65 characters.",
            "accept": [
                "Exactly one <h1> present",
                "Visible text (no display:none)",
                "Aligns with title intent",
            ],
        },
        "canonical_differs": {
            "finding": "Canonical URL differs",
            "severity": "medium",
            "fix": "Update <link rel='canonical'> to match the preferred URL/host; remove conflicting canonicals.",
            "accept": [
                "Canonical equals preferred URL",
                "No cross-host canonical drift",
                "No duplicate canonicals",
            ],
        },
    }
    out: List[Dict[str, Any]] = []
    for p in (crawl.get("pages") or []):
        url = p.get("final_url") or p.get("url")
        for issue in (p.get("issues") or []):
            spec = ISSUE_MAP.get(issue)
            if not spec:
                continue
            out.append(
                {
                    "url": url,
                    "finding": spec["finding"],
                    "severity": spec["severity"],
                    "fix": spec["fix"],
                    "accept": spec["accept"],
                }
            )
    return out


def _aeo_findings_from_faq(faq: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not faq:
        return out
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
    return out


def _geo_recommendations(site_meta: Dict[str, Any], schema_job: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    home = (site_meta.get("url") or "").strip() or "https://example.com/"
    present_type = (schema_job or {}).get("biz_type")
    recs: List[Dict[str, Any]] = []

    recs.append(
        {
            "url": home,
            "finding": "Add Organization + Logo + sameAs (entity grounding)",
            "severity": "high",
            "fix": "Provide Organization/Logo markup and sameAs links to authoritative IDs (LinkedIn, Wikidata).",
            "accept": [
                "Rich Results Test: valid Organization & Logo",
                "Logo URL public & HTTPS",
                "sameAs resolves (2+ authoritative profiles)",
            ],
        }
    )

    recs.append(
        {
            "url": home,
            "finding": "Add WebSite schema with consistent site name",
            "severity": "medium",
            "fix": "Include WebSite JSON-LD and ensure site name consistency across pages & Search Console.",
            "accept": [
                "Rich Results Test: valid WebSite",
                "Site name matches brand guidelines",
            ],
        }
    )

    recs.append(
        {
            "url": home.rstrip("/") + "/faq",
            "finding": "Add BreadcrumbList on FAQ and sections",
            "severity": "medium",
            "fix": "Provide BreadcrumbList JSON-LD reflecting on-page breadcrumbs; keep order & URLs consistent.",
            "accept": [
                "Valid BreadcrumbList",
                "Positions sequential (1..n)",
                "URLs canonical",
            ],
        }
    )

    if present_type and str(present_type).lower() == "faqpage":
        recs.append(
            {
                "url": home,
                "finding": "Broaden schema coverage beyond FAQPage",
                "severity": "medium",
                "fix": "Add Article/BlogPosting where applicable; add Person for authors; link entities with about/mentions.",
                "accept": [
                    "Article/BlogPosting valid on content pages",
                    "Author (Person) present with sameAs where appropriate",
                ],
            }
        )

    return recs


def _code_snippets(site_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pre-baked JSON-LD blocks for Appendix."""
    home = (site_meta.get("url") or "").strip().rstrip("/") or "https://example.com"
    brand = site_meta.get("account_name") or "YourBrand"

    org = {
        "@context": "https://schema.org",
        "@type": "Organization",
        "name": brand,
        "url": f"{home}/",
        "logo": f"{home}/favicon-512.png",
        "sameAs": [
            "https://www.linkedin.com/company/yourbrand",
            "https://www.wikidata.org/wiki/QXXXXX",
        ],
    }

    website = {
        "@context": "https://schema.org",
        "@type": "WebSite",
        "name": brand,
        "url": f"{home}/",
    }

    breadcrumbs = {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": [
            {"@type": "ListItem", "position": 1, "name": "Home", "item": f"{home}/"},
        ],
    }

    return [
        {"title": "Organization + Logo + sameAs", "json": org},
        {"title": "WebSite", "json": website},
        {"title": "BreadcrumbList (example)", "json": breadcrumbs},
    ]


# ---------------------------
# Optional LLM executive summary
# ---------------------------
def _try_llm_summary(site_meta, seo_rows, geo_rows, aeo_rows) -> Optional[str]:
    if not _openai_client:
        return None
    try:
        sys = (
            "You are a senior SEO/GEO/AEO auditor. Produce a crisp executive summary "
            "(2–4 short paragraphs) using only the inputs. Be concrete and prioritise fixes."
        )
        payload = {
            "site": {
                "name": site_meta.get("account_name"),
                "url": site_meta.get("url"),
                "language": site_meta.get("language"),
                "country": site_meta.get("country"),
            },
            "counts": {
                "seo_findings": len(seo_rows),
                "geo_recommendations": len(geo_rows),
                "aeo_findings": len(aeo_rows),
            },
            "highlights": {
                "seo": [r["finding"] for r in seo_rows[:3]],
                "geo": [r["finding"] for r in geo_rows[:3]],
                "aeo": [r["finding"] for r in aeo_rows[:3]],
            },
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


# ---------------------------
# Report generation
# ---------------------------
def generate_report(conn, job):
    site_id = job["site_id"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Inputs
    site_meta = _fetch_site_meta(conn, site_id)
    crawl = _fetch_latest_job(conn, site_id, "crawl")
    keywords = _fetch_latest_job(conn, site_id, "keywords")
    faq = _fetch_latest_job(conn, site_id, "faq")
    schema_job = _fetch_latest_job(conn, site_id, "schema")

    # Build findings
    seo_rows = _seo_findings_from_crawl(crawl)
    aeo_rows = _aeo_findings_from_faq(faq)
    geo_rows = _geo_recommendations(site_meta, schema_job)

    # Optional executive summary
    exec_summary = _try_llm_summary(site_meta, seo_rows, geo_rows, aeo_rows) or (
        "This report lists concrete issues and recommendations across SEO (technical), "
        "GEO (entity/schema), and AEO (answer readiness). Items are prioritised with clear "
        "acceptance criteria so your team can ship fixes confidently."
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
    elems.append(Spacer(1, 10))
    elems.append(Paragraph(exec_summary, S["Small"]))
    elems.append(PageBreak())

    # --- SEO Findings ---
    elems.append(Paragraph("SEO Findings", S["Heading2"]))
    if not seo_rows:
        elems.append(Paragraph("No findings.", S["Normal"]))
    else:
        headers = ["Page URL", "Finding", "Severity", "Fix (summary)", "Acceptance Criteria"]
        # Five-column layout sized for A4 content width
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
    elems.append(Paragraph("GEO Recommendations", S["Heading2"]))
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
    elems.append(Paragraph("AEO Findings", S["Heading2"]))
    if not aeo_rows:
        elems.append(Paragraph("No findings.", S["Normal"]))
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

    # --- Appendix: JSON-LD snippets ---
    elems.append(Paragraph("Appendix — JSON-LD Snippets (paste & adapt)", S["Heading2"]))
    for block in _code_snippets(site_meta):
        elems.append(Paragraph(block["title"], S["H3tight"]))
        pretty = json.dumps(block["json"], indent=2, ensure_ascii=False)
        # Keep code within frame; shrink if needed
        kif = KeepInFrame(doc.width, 120 * mm, [Code(pretty)], mode="shrink")
        elems.append(KeepTogether([kif, Spacer(1, 6)]))

    # If schema job exists, include generated snippet truncated
    if schema_job and schema_job.get("schema"):
        elems.append(Paragraph("Generated (from jobs.schema)", S["H3tight"]))
        pretty = json.dumps(schema_job["schema"], indent=2, ensure_ascii=False)[:4000]
        kif = KeepInFrame(doc.width, 140 * mm, [Code(pretty)], mode="shrink")
        elems.append(KeepTogether([kif, Spacer(1, 6)]))

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
