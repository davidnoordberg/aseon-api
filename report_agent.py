# report_agent.py
import io, base64, json
from datetime import datetime
from psycopg.rows import dict_row
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# --- Helpers ---
def _get_latest_job(conn, site_id, jtype):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT output
              FROM jobs
             WHERE site_id=%s AND type=%s AND status='done'
             ORDER BY finished_at DESC NULLS LAST, created_at DESC
             LIMIT 1
        """, (site_id, jtype))
        r = cur.fetchone()
        return r["output"] if r else None

def _make_html(site, crawl, keywords, faqs, schema) -> str:
    html = [f"<h1>Report — {site['url']}</h1>"]
    html.append(f"<p><b>Generated:</b> {datetime.utcnow().isoformat()} UTC</p>")

    if crawl:
        html.append("<h2>Crawl summary</h2>")
        s = crawl.get("summary", {})
        html.append(f"<p>Pages: {s.get('pages_total')} | 200: {s.get('ok_200')} "
                    f"| 4xx/5xx: {s.get('errors_4xx_5xx')}</p>")
        if crawl.get("quick_wins"):
            html.append("<ul>")
            for q in crawl["quick_wins"]:
                html.append(f"<li>{q.get('type')}</li>")
            html.append("</ul>")

    if keywords:
        html.append("<h2>Keywords</h2>")
        for cluster, items in (keywords.get("clusters") or {}).items():
            html.append(f"<h3>{cluster.title()}</h3><ul>")
            for kw in items[:10]:
                html.append(f"<li>{kw}</li>")
            html.append("</ul>")

    if faqs:
        html.append("<h2>FAQs</h2><ul>")
        for f in faqs.get("faqs", []):
            html.append(f"<li><b>{f.get('q')}</b> — {f.get('a')}</li>")
        html.append("</ul>")

    if schema:
        html.append("<h2>Schema</h2>")
        html.append(f"<pre>{json.dumps(schema.get('schema'), indent=2)}</pre>")

    return "\n".join(html)

def _make_pdf(site, html_text: str) -> str:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    for line in html_text.splitlines():
        if line.startswith("<h1>"):
            story.append(Paragraph(line.strip("<h1></h1>"), styles["Title"]))
        elif line.startswith("<h2>"):
            story.append(Paragraph(line.strip("<h2></h2>"), styles["Heading2"]))
        elif line.startswith("<h3>"):
            story.append(Paragraph(line.strip("<h3></h3>"), styles["Heading3"]))
        elif line.startswith("<li>"):
            story.append(Paragraph(line.strip("<li></li>"), styles["Normal"]))
        else:
            story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 12))
    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return base64.b64encode(pdf_bytes).decode("utf-8")

# --- Main entry ---
def generate_report(conn, job):
    site_id = job["site_id"]
    payload = job.get("payload") or {}

    site = {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT url, language, country FROM sites WHERE id=%s
        """, (site_id,))
        site = cur.fetchone() or {}

    crawl = _get_latest_job(conn, site_id, "crawl")
    keywords = _get_latest_job(conn, site_id, "keywords")
    faqs = _get_latest_job(conn, site_id, "faq")
    schema = _get_latest_job(conn, site_id, "schema")

    html = _make_html(site, crawl, keywords, faqs, schema)

    pdf_b64 = None
    if payload.get("format") in ("pdf", "both"):
        pdf_b64 = _make_pdf(site, html)

    result = {"html": html}
    if pdf_b64:
        result["pdf_base64"] = pdf_b64
    return result
