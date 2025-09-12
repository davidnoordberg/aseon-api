# report_agent.py
import base64
from io import BytesIO
from typing import Dict, Any
from xhtml2pdf import pisa

def _html_to_pdf_bytes(html: str) -> bytes:
    """Render HTML naar PDF en geef bytes terug of raise RuntimeError."""
    result = BytesIO()
    pdf_status = pisa.CreatePDF(html, dest=result)
    if pdf_status.err:
        raise RuntimeError(f"xhtml2pdf failed with {pdf_status.err} errors")
    return result.getvalue()

def generate_report(conn, job: Dict[str, Any]) -> Dict[str, Any]:
    site_id = job["site_id"]
    payload = job.get("payload") or {}
    fmt = (payload.get("format") or "markdown").lower()

    # Minimalistisch HTML rapport
    html = f"""
    <html>
      <head><title>Aseon Report</title></head>
      <body>
        <h1>Aseon Report</h1>
        <p><b>Site ID:</b> {site_id}</p>
        <ul>
          <li>Crawl: OK</li>
          <li>Keywords: OK</li>
          <li>FAQ: OK</li>
          <li>Schema: OK</li>
        </ul>
      </body>
    </html>
    """

    if fmt == "pdf":
        pdf_bytes = _html_to_pdf_bytes(html)
        return {
            "format": "pdf",
            "pdf_base64": base64.b64encode(pdf_bytes).decode("utf-8")
        }
    elif fmt == "html":
        return {"format": "html", "report": html}
    else:
        md = (
            f"# Aseon Report\n\n"
            f"- Site: {site_id}\n"
            f"- Crawl: OK\n"
            f"- Keywords: OK\n"
            f"- FAQ: OK\n"
            f"- Schema: OK"
        )
        return {"format": "markdown", "report": md}
