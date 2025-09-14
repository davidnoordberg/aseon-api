# report_agent.py
# Genereert een PDF-rapport op basis van de laatste job-outputs (crawl, keywords, faq, schema).
# Vereist: psycopg (v3), reportlab. DB-url via env: DATABASE_URL

from __future__ import annotations
import os, io, json, base64, datetime as dt
import psycopg
from typing import Any, Dict, Optional, Tuple

# ReportLab
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem
)


# ---------- DB helpers ----------

def _get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL env var ontbreekt.")
    # psycopg v3: autocommit niet nodig voor read-only
    return psycopg.connect(dsn, application_name="report_agent")

def _fetch_site_url(conn, site_id: str) -> Optional[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT url FROM sites WHERE id=%s LIMIT 1", (site_id,))
        row = cur.fetchone()
        return (row[0] if row else None)

def _fetch_latest_job(conn, site_id: str, jtype: str) -> Optional[Dict[str, Any]]:
    """
    Haal de meest recente succesvolle job van een type op, inclusief output-json.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, type, status, created_at, output
              FROM jobs
             WHERE site_id=%s AND type=%s AND status='done'
             ORDER BY created_at DESC
             LIMIT 1
        """, (site_id, jtype))
        row = cur.fetchone()
        if not row:
            return None
        job = {
            "id": row[0],
            "type": row[1],
            "status": row[2],
            "created_at": row[3].isoformat() if row[3] else None,
            "output": None
        }
        try:
            job["output"] = row[4] if isinstance(row[4], dict) else json.loads(row[4] or "{}")
        except Exception:
            job["output"] = {}
        return job

def _safe_get(d: Dict[str, Any], path: Tuple[str, ...], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ---------- PDF helpers ----------

def _doc_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], fontSize=18, leading=22, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=14, leading=18, spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=12))
    styles.add(ParagraphStyle(name="Mono", parent=styles["BodyText"], fontName="Courier", fontSize=8, leading=10))
    return styles

def _kv_table(pairs):
    tbl = Table(pairs, colWidths=[55*mm, 110*mm])
    tbl.setStyle(TableStyle([
        ("FONT", (0,0), (-1,-1), "Helvetica", 10),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.whitesmoke, colors.white]),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("BOX", (0,0), (-1,-1), 0.5, colors.grey),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    return tbl


# ---------- Main API ----------

def generate_report(site_id: str) -> Dict[str, Any]:
    """
    Bouwt het rapport en retourneert { "format": "pdf", "pdf_base64": "<...>" }.
    Wordt aangeroepen door general_agent bij type='report'.
    """
    # Data verzamelen
    with _get_conn() as conn:
        site_url = _fetch_site_url(conn, site_id) or "(unknown site)"

        crawl_job   = _fetch_latest_job(conn, site_id, "crawl")
        kw_job      = _fetch_latest_job(conn, site_id, "keywords")
        faq_job     = _fetch_latest_job(conn, site_id, "faq")
        schema_job  = _fetch_latest_job(conn, site_id, "schema")

    # Extracten met fallbacks
    crawl_out   = crawl_job["output"]   if crawl_job else {}
    kw_out      = kw_job["output"]      if kw_job else {}
    faq_out     = faq_job["output"]     if faq_job else {}
    schema_out  = schema_job["output"]  if schema_job else {}

    summary     = _safe_get(crawl_out, ("summary",), {}) or {}
    pages       = _safe_get(crawl_out, ("pages",), []) or []

    # KPI’s
    kpi = [
        ("Pages crawled",          str(summary.get("pages_total", 0))),
        ("HTTP 200",               str(summary.get("ok_200", 0))),
        ("3xx redirects",          str(summary.get("redirect_3xx", 0))),
        ("4xx/5xx errors",         str(summary.get("errors_4xx_5xx", 0))),
        ("Fetch errors",           str(summary.get("fetch_errors", 0))),
        ("Duration (ms)",          str(summary.get("duration_ms", 0))),
    ]

    # Quick-wins (van crawl)
    quick_wins = []
    for qw in (crawl_out.get("quick_wins") or []):
        t = qw.get("type")
        if t == "missing_meta_description":
            quick_wins.append("Add concise meta descriptions on key pages.")
        elif t == "missing_h1":
            quick_wins.append("Add a clear on-page H1 heading.")
        elif t == "canonical_differs":
            quick_wins.append("Fix canonical/URL mismatch to avoid index dilution.")
        elif t:
            quick_wins.append(t)

    # Keywords
    kw_list = kw_out.get("keywords") or []
    clusters = kw_out.get("clusters") or {}
    suggestions = kw_out.get("suggestions") or []

    # FAQ
    faqs = faq_out.get("faqs") or []

    # Schema
    schema_snippet = schema_out.get("schema") or {}

    # ---------- PDF bouwen ----------
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16*mm, rightMargin=16*mm, topMargin=16*mm, bottomMargin=16*mm
    )
    styles = _doc_styles()

    story = []
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    story.append(Paragraph("Aseon Report", styles["H1"]))
    story.append(Paragraph(f"Site: {site_url} &nbsp;&nbsp;&nbsp; Site ID: {site_id}", styles["Small"]))
    story.append(Paragraph(f"Generated: {now}", styles["Small"]))
    story.append(Spacer(1, 8))

    # KPI blok
    story.append(Paragraph("Crawl summary", styles["H2"]))
    story.append(_kv_table(kpi))
    story.append(Spacer(1, 8))

    # Quick wins
    story.append(Paragraph("Quick wins", styles["H2"]))
    if quick_wins:
        story.append(ListFlowable([ListItem(Paragraph(q, styles["BodyText"])) for q in quick_wins], bulletType="bullet"))
    else:
        story.append(Paragraph("No immediate issues detected.", styles["BodyText"]))
    story.append(Spacer(1, 10))

    # Top pages (max 8)
    story.append(Paragraph("Pages (sample)", styles["H2"]))
    if pages:
        rows = [["URL", "Title", "Status", "Issues"]]
        for p in pages[:8]:
            issues = ", ".join(p.get("issues") or []) or "-"
            rows.append([p.get("final_url") or p.get("url") or "-", p.get("title") or "-", str(p.get("status", "-")), issues])
        t = Table(rows, colWidths=[70*mm, 60*mm, 15*mm, 35*mm])
        t.setStyle(TableStyle([
            ("FONT", (0,0), (-1,0), "Helvetica-Bold", 10),
            ("FONT", (0,1), (-1,-1), "Helvetica", 9),
            ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
            ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No pages captured yet.", styles["BodyText"]))
    story.append(Spacer(1, 10))

    # Keywords & clusters
    story.append(Paragraph("Keywords", styles["H2"]))
    if kw_list:
        # toon max 20
        shown = kw_list[:20]
        story.append(ListFlowable([ListItem(Paragraph(k, styles["BodyText"])) for k in shown], bulletType="bullet"))
        if len(kw_list) > 20:
            story.append(Paragraph(f"... and {len(kw_list)-20} more", styles["Small"]))
    else:
        story.append(Paragraph("No keyword output available.", styles["BodyText"]))
    story.append(Spacer(1, 6))

    if clusters:
        story.append(Paragraph("Keyword clusters", styles["H2"]))
        for cname, items in clusters.items():
            story.append(Paragraph(f"• <b>{cname}</b>", styles["BodyText"]))
            story.append(ListFlowable([ListItem(Paragraph(i, styles["Small"])) for i in items[:8]], bulletType="bullet"))
        story.append(Spacer(1, 6))

    if suggestions:
        story.append(Paragraph("Suggested pages", styles["H2"]))
        for s in suggestions[:5]:
            title = s.get("page_title") or "Suggested page"
            kws = s.get("grouped_keywords") or []
            story.append(Paragraph(f"• <b>{title}</b>", styles["BodyText"]))
            if kws:
                story.append(ListFlowable([ListItem(Paragraph(k, styles["Small"])) for k in kws[:8]], bulletType="bullet"))
        story.append(Spacer(1, 10))

    # FAQ
    story.append(Paragraph("FAQ (draft)", styles["H2"]))
    if faqs:
        for f in faqs[:8]:
            q = f.get("q") or "-"
            a = f.get("a") or "-"
            story.append(Paragraph(f"<b>Q:</b> {q}", styles["BodyText"]))
            story.append(Paragraph(f"<b>A:</b> {a}", styles["Small"]))
            story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("No FAQ output available.", styles["BodyText"]))
    story.append(Spacer(1, 10))

    # Schema
    story.append(Paragraph("Schema (JSON-LD preview)", styles["H2"]))
    if schema_snippet:
        try:
            pretty = json.dumps(schema_snippet, ensure_ascii=False, indent=2)[:3000]
        except Exception:
            pretty = str(schema_snippet)[:3000]
        story.append(Paragraph(f"<font name='Courier'>{pretty.replace(' ', '&nbsp;').replace('<','&lt;').replace('>','&gt;')}</font>", styles["Mono"]))
    else:
        story.append(Paragraph("No schema output available.", styles["BodyText"]))

    # Render PDF -> base64
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")

    return {"format": "pdf", "pdf_base64": pdf_b64}
