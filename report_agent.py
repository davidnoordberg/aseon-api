# report_agent.py
import os, json, io, datetime
import psycopg
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

DATABASE_URL = os.environ["DATABASE_URL"]

def _get_latest_job(conn, site_id, jtype):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT output
            FROM jobs
            WHERE site_id=%s AND type=%s AND status='done'
            ORDER BY created_at DESC
            LIMIT 1
        """, (site_id, jtype))
        r = cur.fetchone()
        return (r[0] if r and r[0] else None)

def _score_workload(crawl, kw):
    pages = (crawl or {}).get("summary",{}).get("pages_total",0)
    issues = sum(len(p.get("issues",[])) for p in (crawl or {}).get("pages",[]))
    kws = len((kw or {}).get("keywords",[]))
    # simpele mapping naar pakket
    if pages >= 400 or kws >= 180 or issues >= 120:
        plan = "Premium"   # 8 pages/mo, 200 seeds, 10 languages, advanced tests
    elif pages >= 120 or kws >= 90 or issues >= 50:
        plan = "Standard"  # 4 pages/mo, 100 seeds, 5 languages
    else:
        plan = "Basic"     # 2 pages/mo, 50 seeds, 2 languages
    return plan, {"pages":pages,"issues":issues,"keywords":kws}

def _para(text, styles):
    return Paragraph(text, styles["BodyText"])

def _h(text, styles):
    return Paragraph(f"<b>{text}</b>", styles["Heading2"])

def _table(data, colWidths=None):
    t = Table(data, colWidths=colWidths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), colors.HexColor("#f0f3f7")),
        ('TEXTCOLOR',(0,0),(-1,0), colors.HexColor("#333333")),
        ('LINEBELOW',(0,0),(-1,0), 0.5, colors.HexColor("#d9e1ec")),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [colors.white, colors.HexColor("#fafbfd")]),
        ('GRID',(0,0),(-1,-1), 0.25, colors.HexColor("#eaeef5")),
        ('LEFTPADDING',(0,0),(-1,-1),6),
        ('RIGHTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),4),
        ('BOTTOMPADDING',(0,0),(-1,-1),4),
    ]))
    return t

def build_pdf(site_id: str) -> bytes:
    conn = psycopg2.connect(DATABASE_URL)
    try:
        crawl = _get_latest_job(conn, site_id, "crawl") or {}
        kw    = _get_latest_job(conn, site_id, "keywords") or {}
        faq   = _get_latest_job(conn, site_id, "faq") or {}
        schema= _get_latest_job(conn, site_id, "schema") or {}

    finally:
        conn.close()

    styles = getSampleStyleSheet()
    story = []
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Titel
    story.append(Paragraph("<b>Aseon — AI Visibility Report</b>", styles["Title"]))
    story.append(_para(f"Site ID: {site_id} &nbsp;&nbsp;•&nbsp;&nbsp; Generated: {now}", styles))
    story.append(Spacer(1, 0.4*cm))

    # Executive summary + pakket
    plan, metrics = _score_workload(crawl, kw)
    story.append(_h("Executive summary", styles))
    bullets = [
        f"Detected pages crawled: <b>{metrics['pages']}</b>",
        f"Total issues (h1/meta/canonical etc.): <b>{metrics['issues']}</b>",
        f"Keywords found/suggested: <b>{metrics['keywords']}</b>",
        f"Recommended plan: <b>{plan}</b> (matching your product tiers)"
    ]
    story.append(_para("• " + "<br/>• ".join(bullets), styles))
    story.append(Spacer(1, 0.3*cm))

    # Crawl metrics
    csum = (crawl or {}).get("summary",{})
    story.append(_h("Crawl overview", styles))
    data = [
        ["Metric","Value"],
        ["Pages total", csum.get("pages_total",0)],
        ["HTTP 200", csum.get("ok_200",0)],
        ["3xx redirects", csum.get("redirect_3xx",0)],
        ["4xx/5xx errors", csum.get("errors_4xx_5xx",0)],
        ["Avg TTFB (ms)", csum.get("avg_ttfb_ms",0)],
        ["Avg size (bytes)", csum.get("avg_bytes",0)],
    ]
    story.append(_table(data, colWidths=[6*cm, 9*cm]))
    story.append(Spacer(1, 0.3*cm))

    # Top issues per pagina (max 10)
    pages = (crawl or {}).get("pages",[])[:10]
    if pages:
        story.append(_h("Top issues per page (sample)", styles))
        rows = [["URL","Issues"]]
        for p in pages:
            iss = ", ".join(p.get("issues",[]) or ["—"])
            rows.append([p.get("final_url") or p.get("url"), iss])
        story.append(_table(rows, colWidths=[10*cm, 5*cm]))
        story.append(Spacer(1, 0.3*cm))

    # Keywords samenvatting
    if kw:
        story.append(_h("Keyword opportunities (clusters)", styles))
        clusters = (kw.get("clusters") or {})
        rows = [["Cluster","Examples (up to 5)"]]
        for k in ("transactional","informational","navigational"):
            ex = ", ".join((clusters.get(k) or [])[:5]) or "—"
            rows.append([k.capitalize(), ex])
        story.append(_table(rows, colWidths=[5*cm, 10*cm]))
        story.append(Spacer(1, 0.3*cm))

    # FAQ (sample)
    faqs = (faq or {}).get("faqs") or []
    if faqs:
        story.append(_h("Proposed FAQs (sample)", styles))
        rows = [["Q","A (short)"]]
        for f in faqs[:5]:
            rows.append([f.get("q","—"), (f.get("a","—")[:200] + ("…" if len(f.get("a",""))>200 else ""))])
        story.append(_table(rows, colWidths=[7*cm, 8*cm]))
        story.append(Spacer(1, 0.3*cm))

    # Schema snippet
    if schema and schema.get("schema"):
        story.append(_h("Schema example (FAQPage)", styles))
        code = json.dumps(schema["schema"], ensure_ascii=False, indent=2)
        story.append(_para(f"<font face='Courier'>{code.replace(' ', '&nbsp;').replace('\n','<br/>')}</font>", styles))
        story.append(Spacer(1, 0.3*cm))

    # Prioriteiten (quick wins)
    qwins = (crawl or {}).get("quick_wins") or []
    if qwins:
        story.append(_h("Top quick wins", styles))
        items = ", ".join(sorted(set(q['type'] for q in qwins)))
        story.append(_para(f"• {items}", styles))

    # Pakket-mapping uitleg
    story.append(PageBreak())
    story.append(_h("Plan mapping (Basic / Standard / Premium)", styles))
    story.append(_para(
        "We map crawl size, issue complexity and keyword workload to your plans:\n"
        "- Basic: up to ~100 pages audit / ~50 issues / ~50 seeds\n"
        "- Standard: up to ~500 pages audit / ~120 issues / ~100 seeds\n"
        "- Premium: up to ~1,000 pages audit / ~200 seeds / advanced testing\n", styles))

    # PDF render
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.2*cm, leftMargin=1.5*cm, rightMargin=1.5*cm, bottomMargin=1.2*cm)
    doc.build(story)
    return buf.getvalue()
