# report_agent.py
import base64
import io
from datetime import datetime, timezone

from psycopg.rows import dict_row
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

def _fetch_latest_job(conn, site_id, jtype):
    """Haal de laatste succesvolle job-output op van een bepaald type."""
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


def generate_report(conn, job):
    """
    Bouw een PDF-rapport met crawl/faq/schema samenvatting.
    return: dict {"pdf_base64": "...", "meta": {...}}
    """
    site_id = job["site_id"]
    payload = job.get("payload", {}) or {}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # haal laatste resultaten op
    crawl = _fetch_latest_job(conn, site_id, "crawl")
    faq = _fetch_latest_job(conn, site_id, "faq")
    schema = _fetch_latest_job(conn, site_id, "schema")
    keywords = _fetch_latest_job(conn, site_id, "keywords")

    # PDF opbouwen
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elems = []

    elems.append(Paragraph("ASEON Site Report", styles["Title"]))
    elems.append(Paragraph(f"Generated at: {now}", styles["Normal"]))
    elems.append(Spacer(1, 20))

    # --- Crawl summary ---
    if crawl:
        summary = crawl.get("summary", {})
        elems.append(Paragraph("Crawl Summary", styles["Heading2"]))
        elems.append(
            Paragraph(
                f"Pages crawled: {summary.get('pages_total', 0)} "
                f"(200 OK: {summary.get('ok_200', 0)}, "
                f"Errors: {summary.get('errors_4xx_5xx', 0)})",
                styles["Normal"],
            )
        )
        if crawl.get("quick_wins"):
            elems.append(Paragraph("Quick Wins:", styles["Heading3"]))
            for win in crawl["quick_wins"]:
                elems.append(Paragraph(f"- {win['type']}", styles["Normal"]))
        elems.append(Spacer(1, 15))
        elems.append(PageBreak())

    # --- Keywords ---
    if keywords:
        elems.append(Paragraph("Keyword Suggestions", styles["Heading2"]))
        kws = keywords.get("keywords", [])
        for kw in kws[:30]:
            elems.append(Paragraph(f"- {kw}", styles["Normal"]))
        elems.append(PageBreak())

    # --- FAQ ---
    if faq and "faqs" in faq:
        elems.append(Paragraph("Frequently Asked Questions", styles["Heading2"]))
        for item in faq["faqs"]:
            q = item.get("q")
            a = item.get("a")
            if q and a:
                elems.append(Paragraph(f"Q: {q}", styles["Heading3"]))
                elems.append(Paragraph(f"A: {a}", styles["Normal"]))
                elems.append(Spacer(1, 10))
        elems.append(PageBreak())

    # --- Schema ---
    if schema:
        elems.append(Paragraph("Schema.org Snippet", styles["Heading2"]))
        elems.append(Paragraph("Type: " + str(schema.get("biz_type")), styles["Normal"]))
        sc = schema.get("schema")
        if sc:
            import json

            json_str = json.dumps(sc, indent=2)
            # toon max 1000 chars
            elems.append(Paragraph(f"<pre>{json_str[:1000]}</pre>", styles["Code"]))
        elems.append(PageBreak())

    # fallback als leeg
    if not elems:
        elems.append(Paragraph("No data available for this site.", styles["Normal"]))

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
                "crawl": bool(crawl),
                "keywords": bool(keywords),
                "faq": bool(faq),
                "schema": bool(schema),
            },
        },
    }
