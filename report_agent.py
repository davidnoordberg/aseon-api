# report_agent.py
import os, json, base64
from io import BytesIO
from xhtml2pdf import pisa

# Dummy: hier zou je je echte bundeling doen (crawl, keywords, faq, schema, quick wins, plan)
def _assemble_html(site_info: dict, sections: dict) -> str:
    # Minimale HTML layout
    html = f"""
    <html>
      <head>
        <meta charset="utf-8"/>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 40px; }}
          h1 {{ color: #2c3e50; }}
          h2 {{ color: #34495e; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
          pre {{ background: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; }}
          .section {{ margin-bottom: 30px; }}
        </style>
      </head>
      <body>
        <h1>SEO Report for {site_info.get("url")}</h1>
        <p><b>Language:</b> {site_info.get("language")}<br/>
        <b>Country:</b> {site_info.get("country")}</p>
    """
    for name, content in sections.items():
        html += f"<div class='section'><h2>{name.title()}</h2>"
        if isinstance(content, dict) or isinstance(content, list):
            html += f"<pre>{json.dumps(content, indent=2, ensure_ascii=False)}</pre>"
        else:
            html += f"<p>{content}</p>"
        html += "</div>"
    html += "</body></html>"
    return html

def _html_to_pdf_bytes(html: str) -> bytes:
    result = BytesIO()
    pisa.CreatePDF(html, dest=result)
    return result.getvalue()

def generate_report(conn, job: dict) -> dict:
    site_id = job["site_id"]
    payload = job.get("payload") or {}
    fmt = (payload.get("format") or "both").lower()

    # haal site info uit DB
    with conn.cursor() as cur:
        cur.execute("SELECT url, language, country FROM sites WHERE id=%s", (site_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError("Site not found")
        site_info = {"url": row[0], "language": row[1], "country": row[2]}

    # dummy sections (hier kun je get_latest_job_output(conn, site_id, "crawl") enz. doen)
    sections = {
        "crawl": {"status": "ok", "pages": 5},
        "keywords": ["seo tips", "fast api", "pdf export"],
        "faq": [{"q":"Wat is SEO?","a":"Zoekmachineoptimalisatie."}],
        "schema": {"@context":"https://schema.org","@type":"Organization","name":"Test"},
        "quick_wins": ["Add meta description", "Fix missing H1"],
        "plan": "Focus on content clustering + schema markup."
    }

    # Bouw HTML
    html = _assemble_html(site_info, sections)

    out = {}
    if fmt in ("html","both"):
        out["html"] = html
    if fmt in ("pdf","both"):
        pdf_bytes = _html_to_pdf_bytes(html)
        out["pdf_base64"] = base64.b64encode(pdf_bytes).decode("utf-8")

    return out
