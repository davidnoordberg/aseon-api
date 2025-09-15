# report_agent.py
import base64
import io
import json
import os
import re
from datetime import datetime, timezone
from psycopg.rows import dict_row
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

from openai import OpenAI
from rag_helper import get_rag_context

OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TIMEOUT_SEC = 30

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def _fetch_latest_job(conn, site_id, jtype):
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

def _ai_summarize(site_id, crawl, keywords, faq, schema, conn):
    ctx = get_rag_context(conn, site_id=site_id, query="site audit", kb_tags=["SEO","Schema","AEO","Quality"])
    crawl_txt = json.dumps(crawl or {}, indent=2)
    kw_txt = json.dumps(keywords or {}, indent=2)
    faq_txt = json.dumps(faq or {}, indent=2)
    schema_txt = json.dumps(schema or {}, indent=2)

    sys = (
        "You are an SEO + AEO auditor. Write a professional audit report. "
        "Use crawl, keywords, FAQ, and schema data plus KB context. "
        "Highlight strengths, weaknesses, and give prioritized recommendations. "
        "Include an overall Executive Summary, a Scoring Matrix (1–10) for: "
        "Crawl health, Content quality, Schema coverage, AEO readiness. "
        "At the end, output an Overall Site Health Score (0–100) = average of the four subscores * 10. "
        "Format clearly in Markdown."
    )
    user = f"""
Crawl data:
{crawl_txt}

Keywords:
{kw_txt}

FAQs:
{faq_txt}

Schema:
{schema_txt}

--- SITE CONTEXT ---
{ctx.get("site_ctx")}

--- KB CONTEXT ---
{ctx.get("kb_ctx")}
"""

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        temperature=0.3,
        timeout=OPENAI_TIMEOUT_SEC,
    )
    return resp.choices[0].message.content.strip()

def _extract_overall_score(ai_text: str) -> int:
    m = re.search(r"(\d{2,3})\s*/\s*100", ai_text)
    if m:
        try:
            val = int(m.group(1))
            return max(0, min(100, val))
        except Exception:
            return None
    return None

def generate_report(conn, job):
    site_id = job["site_id"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    crawl = _fetch_latest_job(conn, site_id, "crawl")
    faq = _fetch_latest_job(conn, site_id, "faq")
    schema = _fetch_latest_job(conn, site_id, "schema")
    keywords = _fetch_latest_job(conn, site_id, "keywords")

    ai_text = _ai_summarize(site_id, crawl, keywords, faq, schema, conn)
    overall_score = _extract_overall_score(ai_text)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elems = []

    elems.append(Paragraph("ASEON Site Report", styles["Title"]))
    elems.append(Paragraph(f"Generated at: {now}", styles["Normal"]))
    if overall_score is not None:
        elems.append(Paragraph(f"Overall Site Health Score: {overall_score}/100", styles["Heading2"]))
    elems.append(Spacer(1, 20))

    # Executive Summary + Scores
    elems.append(Paragraph("Executive Summary & Scores", styles["Heading2"]))
    for line in ai_text.split("\n"):
        if not line.strip():
            elems.append(Spacer(1, 10))
            continue
        elems.append(Paragraph(line, styles["Normal"]))
    elems.append(PageBreak())

    # Appendix
    def add_json_section(title, data):
        if not data: 
            return
        elems.append(Paragraph(title, styles["Heading2"]))
        txt = json.dumps(data, indent=2)
        for chunk in [txt[i:i+1000] for i in range(0, len(txt), 1000)]:
            elems.append(Paragraph(f"<pre>{chunk}</pre>", styles["Code"]))
        elems.append(PageBreak())

    add_json_section("Crawl Data", crawl)
    add_json_section("Keyword Data", keywords)
    add_json_section("FAQ Data", faq)
    add_json_section("Schema Data", schema)

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
                "ai_summary": True,
                "scoring_matrix": True,
                "overall_score": overall_score,
            },
        },
    }
