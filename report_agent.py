# report_agent.py
import io, base64, json
from datetime import datetime
from psycopg.rows import dict_row
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem

# Helper: laatste job-output ophalen
def get_latest_output(conn, site_id, jtype):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT output
              FROM jobs
             WHERE site_id=%s AND type=%s AND status='done'
             ORDER BY finished_at DESC NULLS LAST, created_at DESC
             LIMIT 1
        """, (site_id, jtype))
        row = cur.fetchone()
        return row["output"] if row else None

# Helper: pdf maken
def build_pdf(site_id, crawl, keywords, faq, schema) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    elems = []

    elems.append(Paragraph("<b>Aseon Report</b>", styles["Title"]))
    elems.append(Spacer(1, 12))
    elems.append(Paragraph(f"<b>Site ID:</b> {site_id}", styles["Normal"]))
    elems.append(Spacer(1, 12))

    # Crawl
    if crawl:
        elems.append(Paragraph("<b>Crawl Summary</b>", styles["Heading2"]))
        summ = crawl.get("summary", {})
        elems.append(ListFlowable([
            ListItem(Paragraph(f"Pages total: {summ.get('pages_total')}", styles["Normal"])),
            ListItem(Paragraph(f"OK 200: {summ.get('ok_200')}", styles["Normal"])),
            ListItem(Paragraph(f"Errors: {summ.get('errors_4xx_5xx')}", styles["Normal"]))
        ]))
        elems.append(Spacer(1, 12))

    # Keywords
    if keywords:
        elems.append(Paragraph("<b>Keywords</b>", styles["Heading2"]))
        kws = keywords.get("keywords", [])[:10]
        for kw in kws:
            elems.append(Paragraph(f"- {kw}", styles["Normal"]))
        elems.append(Spacer(1, 12))

    # FAQ
    if faq:
        elems.append(Paragraph("<b>FAQs</b>", styles["Heading2"]))
        faqs = faq.get("faqs", [])[:5]
        for f in faqs:
            elems.append(Paragraph(f"Q: {f.get('q')}", styles["Normal"]))
            elems.append(Paragraph(f"A: {f.get('a')}", styles["Normal"]))
            elems.append(Spacer(1, 6))

    # Schema
    if schema:
        elems.append(Paragraph("<b>Schema.org</b>", styles["Heading2"]))
        schema_json = json.dumps(schema.get("schema", {}), indent=2)
        elems.append(Paragraph(f"<font size=8><pre>{schema_json}</pre></font>", styles["Normal"]))

    doc.build(elems)
    return buf.getvalue()

# Main entrypoint voor general_agent
def generate_report(conn, job):
    site_id = job["site_id"]
    payload = job.get("payload") or {}
    fmt = payload.get("format", "pdf")

    crawl    = get_latest_output(conn, site_id, "crawl")
    keywords = get_latest_output(conn, site_id, "keywords")
    faq      = get_latest_output(conn, site_id, "faq")
    schema   = get_latest_output(conn, site_id, "schema")

    if fmt == "pdf":
        pdf_bytes = build_pdf(site_id, crawl, keywords, faq, schema)
        return {
            "format": "pdf",
            "pdf_base64": base64.b64encode(pdf_bytes).decode("utf-8")
        }
    else:
        return {
            "format": "json",
            "crawl": crawl,
            "keywords": keywords,
            "faq": faq,
            "schema": schema
        }
