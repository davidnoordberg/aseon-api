# report_agent.py
import base64, json
from datetime import datetime
from io import BytesIO
from psycopg.rows import dict_row
from xhtml2pdf import pisa


# ---------- helpers ----------
def _html_to_pdf_bytes(html: str) -> bytes:
    result = BytesIO()
    pdf_status = pisa.CreatePDF(html, dest=result)
    if pdf_status.err:
        raise RuntimeError(f"xhtml2pdf failed with {pdf_status.err} errors")
    return result.getvalue()


def _to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


# ---------- fetchers ----------
def _get_latest_job(conn, site_id: str, jtype: str):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, output, finished_at
              FROM jobs
             WHERE site_id=%s AND type=%s AND status='done'
          ORDER BY COALESCE(finished_at, created_at) DESC
             LIMIT 1
        """,
            (site_id, jtype),
        )
        return cur.fetchone()


# ---------- generator ----------
def generate_report(conn, job: dict) -> dict:
    site_id = job["site_id"]
    payload = job.get("payload") or {}
    fmt = (payload.get("format") or "html").lower()

    # site info
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
        site = cur.fetchone()

    crawl = _get_latest_job(conn, site_id, "crawl")
    keywords = _get_latest_job(conn, site_id, "keywords")
    faq = _get_latest_job(conn, site_id, "faq")
    schema = _get_latest_job(conn, site_id, "schema")

    # HTML rapport
    html_parts = []
    html_parts.append("<h1>Aseon Report</h1>")
    if site:
        html_parts.append(
            f"<p><b>Site:</b> {site['url']}<br>"
            f"<b>Account:</b> {site['account_name']}<br>"
            f"<b>Taal/Land:</b> {site['language']}/{site['country']}<br>"
            f"<small>Generated {datetime.utcnow().isoformat()}</small></p>"
        )

    if crawl and crawl.get("output"):
        c = crawl["output"]
        html_parts.append("<h2>Crawl summary</h2>")
        html_parts.append(
            f"<p>Pages: {c['summary']['pages_total']} | "
            f"200={c['summary']['ok_200']} | "
            f"4xx/5xx={c['summary']['errors_4xx_5xx']}</p>"
        )

    if keywords and keywords.get("output"):
        k = keywords["output"]
        html_parts.append("<h2>Keyword clusters</h2>")
        html_parts.append(f"<pre>{json.dumps(k, indent=2, ensure_ascii=False)}</pre>")

    if faq and faq.get("output"):
        f = faq["output"]
        html_parts.append("<h2>FAQs</h2><ul>")
        for qa in f.get("faqs", []):
            html_parts.append(
                f"<li><b>{qa['q']}</b> — {qa['a']} "
                f"({qa.get('source') or 'no source'})</li>"
            )
        html_parts.append("</ul>")

    if schema and schema.get("output"):
        s = schema["output"]
        html_parts.append("<h2>Schema</h2>")
        html_parts.append(f"<pre>{json.dumps(s, indent=2, ensure_ascii=False)}</pre>")

    html = "<html><body>" + "\n".join(html_parts) + "</body></html>"

    # output
    if fmt == "html":
        return {"format": "html", "html": html}
    elif fmt == "pdf":
        pdf_bytes = _html_to_pdf_bytes(html)
        return {"format": "pdf", "pdf_base64": _to_base64(pdf_bytes)}
    else:
        raise ValueError(f"Unknown format: {fmt}")
