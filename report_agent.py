# report_agent.py
import os, io, re, json, base64
from datetime import datetime
from typing import Dict, Any

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet

styles = getSampleStyleSheet()

def _strip_html(text: str) -> str:
    """Verwijder HTML-tags en hou alleen platte tekst over"""
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text)

def _split_blocks(report_text: str):
    """
    Splits report text in gewone paragrafen en codeblokken
    - Codeblokken: ```...``` of <pre>...</pre>
    - Rest: platte tekst
    """
    blocks = []
    code_pattern = re.compile(r"```(.*?)```", re.S)
    pos = 0
    for m in code_pattern.finditer(report_text):
        if m.start() > pos:
            blocks.append(("text", report_text[pos:m.start()]))
        blocks.append(("code", m.group(1)))
        pos = m.end()
    if pos < len(report_text):
        blocks.append(("text", report_text[pos:]))
    return blocks

def _make_pdf(report_text: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    story = []

    blocks = _split_blocks(report_text)

    for kind, content in blocks:
        if kind == "text":
            text = _strip_html(content).strip()
            if text:
                story.append(Paragraph(text, styles["Normal"]))
                story.append(Spacer(1, 12))
        elif kind == "code":
            code = content.strip()
            story.append(Preformatted(code, styles["Code"]))
            story.append(Spacer(1, 12))

    doc.build(story)
    return buf.getvalue()

def generate_report(conn, job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Genereert een rapport in PDF of HTML/Markdown
    """
    site_id = job["site_id"]
    payload = job.get("payload") or {}
    fmt = payload.get("format", "markdown")

    # Voor nu: dummy inhoud — kan uitgebreid worden met echte data uit crawl/faq/etc.
    report_md = f"""# SEO Report
Site ID: {site_id}
Generated: {datetime.utcnow().isoformat()}

## Crawl
- Crawl OK

## Keywords
- Keyword cluster OK

## FAQ
- FAQ OK

## Schema
- Schema OK
"""
    report_html = f"<h1>SEO Report</h1><p>Site ID: {site_id}</p><p>Generated: {datetime.utcnow().isoformat()}</p>"

    out: Dict[str, Any] = {
        "site_id": site_id,
        "generated_at": datetime.utcnow().isoformat(),
        "format": fmt,
        "report_md": report_md,
        "report_html": report_html,
        "_aseon": {"source": "report_agent.py", "version": "2.0"},
    }

    if fmt == "pdf" or fmt == "both":
        pdf_bytes = _make_pdf(report_md)
        out["pdf_base64"] = base64.b64encode(pdf_bytes).decode("utf-8")

    return out
