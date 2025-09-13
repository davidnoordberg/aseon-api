# report_agent.py
import io
import json
import base64
import datetime as dt
from collections import Counter, defaultdict

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem
from reportlab.lib import colors

def _safe(s):
    if s is None:
        return ""
    # heel simpele ontsmetting – paraparser veilig
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _top_issues(pages, top_n=10):
    c = Counter()
    for p in pages:
        for i in (p.get("issues") or []):
            c[i] += 1
    return c.most_common(top_n)

def _examples_by_issue(pages, issue, max_examples=6):
    ex = []
    for p in pages:
        if issue in (p.get("issues") or []):
            ex.append((p.get("final_url") or p.get("url") or "", _safe(p.get("title") or "")))
        if len(ex) >= max_examples:
            break
    return ex

def _kv_table(data, col1="Metric", col2="Value"):
    rows = [[col1, col2]] + [[_safe(k), _safe(v)] for k,v in data]
    t = Table(rows, colWidths=[70*mm, 90*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f0f2f5")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.black),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#d9dde3")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#fbfbfc")]),
    ]))
    return t

def _bullets(items):
    styles = getSampleStyleSheet()
    return ListFlowable(
        [ListItem(Paragraph(_safe(x), styles["Normal"]), leftIndent=6) for x in items],
        bulletType="bullet",
        start=None
    )

def build_report_pdf(site_id, crawl_output=None, keywords_output=None, faq_output=None, schema_output=None):
    pages = (crawl_output or {}).get("pages") or []
    summary = (crawl_output or {}).get("summary") or {}
    quick_wins = (crawl_output or {}).get("quick_wins") or []

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], fontSize=16, leading=20, spaceAfter=6))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=12, leading=15, spaceAfter=4))
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=9, leading=12))

    flow = []
    flow.append(Paragraph(f"Aseon Site Report", styles["H1"]))
    flow.append(Paragraph(f"Site ID: {_safe(site_id)} &nbsp;&nbsp;|&nbsp;&nbsp; Generated: {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Small"]))
    flow.append(Spacer(1, 6))

    # Summary
    srows = [
        ("Pages crawled", str(summary.get("pages_total", 0))),
        ("HTTP 200", str(summary.get("ok_200", 0))),
        ("3xx redirects", str(summary.get("redirect_3xx", 0))),
        ("4xx/5xx", str(summary.get("errors_4xx_5xx", 0))),
        ("Duration (ms)", str(summary.get("duration_ms", 0))),
        ("Capped by max_pages", "Yes" if summary.get("capped_by_runtime") else "No"),
    ]
    flow.append(Paragraph("Summary", styles["H2"]))
    flow.append(_kv_table(srows))
    flow.append(Spacer(1, 8))

    # Quick wins
    if quick_wins:
        flow.append(Paragraph("Quick wins", styles["H2"]))
        flow.append(_bullets([qw.get("type","") for qw in quick_wins]))
        flow.append(Spacer(1, 8))

    # Top issues
    if pages:
        ti = _top_issues(pages, top_n=10)
        if ti:
            flow.append(Paragraph("Top issues", styles["H2"]))
            flow.append(_kv_table([(k, v) for k,v in ti], "Issue", "Count"))
            flow.append(Spacer(1, 6))

            # voorbeelden per top issue
            for issue, _count in ti[:5]:
                ex = _examples_by_issue(pages, issue, max_examples=6)
                if not ex: 
                    continue
                flow.append(Paragraph(f"Examples – {_safe(issue)}", styles["Small"]))
                flow.append(_kv_table(ex, "URL", "Title"))
                flow.append(Spacer(1, 6))

    # Top pages (first 10)
    if pages:
        flow.append(Paragraph("Key pages (sample)", styles["H2"]))
        sample = []
        for p in pages[:10]:
            sample.append((p.get("final_url") or p.get("url") or "", (p.get("title") or "")[:120]))
        flow.append(_kv_table(sample, "URL", "Title"))
        flow.append(Spacer(1, 8))

    # Keywords (if available)
    if keywords_output:
        flow.append(Paragraph("Keyword suggestions (sample)", styles["H2"]))
        kws = (keywords_output or {}).get("keywords") or []
        flow.append(_bullets(kws[:20]))
        flow.append(Spacer(1, 8))

    # FAQ (if available)
    if faq_output:
        flow.append(Paragraph("Generated FAQs (sample)", styles["H2"]))
        faqs = (faq_output or {}).get("faqs") or []
        blist = [f"Q: {f.get('q','')} — A: {f.get('a','')}" for f in faqs[:6]]
        flow.append(_bullets(blist))
        flow.append(Spacer(1, 8))

    # Schema (if available)
    if schema_output:
        flow.append(Paragraph("Schema (snippet)", styles["H2"]))
        snippet = json.dumps(schema_output.get("schema", {}), ensure_ascii=False)[:1500]
        flow.append(Paragraph(_safe(snippet), styles["Small"]))
        flow.append(Spacer(1, 6))

    doc.build(flow)
    pdf_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"format":"pdf", "pdf_base64": pdf_b64}
