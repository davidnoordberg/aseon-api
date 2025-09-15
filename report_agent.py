import base64
import io
import json
from datetime import datetime, timezone

from psycopg.rows import dict_row
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
)
from reportlab.lib import colors

from openai import OpenAI
from rag_helper import get_rag_context

OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TIMEOUT_SEC = 30
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# -----------------------------------------------------
# DB helpers
# -----------------------------------------------------
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


def _fetch_site_meta(conn, site_id):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT s.url, s.language, s.country, a.name AS account_name
              FROM sites s
              JOIN accounts a ON a.id = s.account_id
             WHERE s.id = %s
            """,
            (site_id,),
        )
        return cur.fetchone() or {}


# -----------------------------------------------------
# LLM synthesis
# -----------------------------------------------------
def _safe(obj, maxlen=4000):
    try:
        s = json.dumps(obj) if not isinstance(obj, str) else obj
        return s[:maxlen]
    except Exception:
        return str(obj)[:maxlen]


def _llm_synthesis(site_meta, crawl, keywords, faq, schema, rag):
    sys = (
        "You are an expert SEO/GEO/AEO auditor. "
        "Synthesize findings into crisp, actionable recommendations. "
        "Score each pillar on 1–10. Be specific, verifiable, no fluff."
    )

    user = f"""
SITE:
- name: {site_meta.get('account_name') or 'Site'}
- url: {site_meta.get('url') or ''} ({site_meta.get('language') or ''}-{site_meta.get('country') or ''})

INPUTS (summaries; truncate applied):
- CRAWL: { _safe(crawl, 4000) if crawl else "null" }
- KEYWORDS: { _safe(keywords, 4000) if keywords else "null" }
- FAQ: { _safe(faq, 4000) if faq else "null" }
- SCHEMA: { _safe(schema, 4000) if schema else "null" }

KB CONTEXT (for policy/best-practice grounding):
{ rag.get('kb_ctx') or '' }

Return STRICT JSON with:
{{
  "summary": {{
    "title": "SEO • GEO • AEO Audit for <site>",
    "executive": "2–4 paragraphs big picture",
    "scores": {{
      "seo": {{
        "technical_health": 0,
        "content_quality": 0,
        "crawl_indexability": 0
      }},
      "geo": {{
        "entity_schema_coverage": 0,
        "eeat_authority": 0
      }},
      "aeo": {{
        "answer_readiness": 0,
        "citation_readiness": 0
      }},
      "overall_score": 0
    }}
  }},
  "prioritized_actions": [
    {{
      "title": "...",
      "why_it_matters": "...",
      "impact": "high|medium|low",
      "effort": "low|medium|high",
      "owner_hint": "content|dev|seo",
      "acceptance_criteria": ["measurable outcome 1","..."]
    }}
  ],
  "key_risks": ["...", "..."]
}}
Rules:
- Use ONLY info derivable from inputs/KB. If unknown, omit.
- Max 10 actions. Keep why_it_matters short and concrete.
"""

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        timeout=OPENAI_TIMEOUT_SEC,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        return data
    except Exception:
        return {
            "summary": {
                "title": "Audit",
                "executive": "LLM synthesis failed",
                "scores": {"seo": {}, "geo": {}, "aeo": {}, "overall_score": 0},
            },
            "prioritized_actions": [],
            "key_risks": [],
        }


# -----------------------------------------------------
# Report generation
# -----------------------------------------------------
def generate_report(conn, job):
    site_id = job["site_id"]
    payload = job.get("payload", {}) or {}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    crawl = _fetch_latest_job(conn, site_id, "crawl")
    faq = _fetch_latest_job(conn, site_id, "faq")
    schema = _fetch_latest_job(conn, site_id, "schema")
    keywords = _fetch_latest_job(conn, site_id, "keywords")
    site_meta = _fetch_site_meta(conn, site_id)

    rag = get_rag_context(conn, site_id=site_id, query="site audit baseline")

    synth = _llm_synthesis(site_meta, crawl, keywords, faq, schema, rag)

    # PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Code", fontName="Courier", fontSize=8))
    elems = []

    elems.append(Paragraph(synth["summary"]["title"], styles["Title"]))
    elems.append(Paragraph(f"Generated at: {now}", styles["Normal"]))
    elems.append(Spacer(1, 12))
    elems.append(Paragraph(synth["summary"]["executive"], styles["Normal"]))
    elems.append(PageBreak())

    # Score table
    scores = synth["summary"]["scores"]
    seo = scores.get("seo", {}) or {}
    geo = scores.get("geo", {}) or {}
    aeo = scores.get("aeo", {}) or {}

    data = [
        ["Pillar / Metric", "Score (1–10)"],
        ["SEO — Technical Health", seo.get("technical_health", 0)],
        ["SEO — Content Quality", seo.get("content_quality", 0)],
        ["SEO — Crawl & Indexability", seo.get("crawl_indexability", 0)],
        ["GEO — Entity/Schema Coverage", geo.get("entity_schema_coverage", 0)],
        ["GEO — E-E-A-T / Authority", geo.get("eeat_authority", 0)],
        ["AEO — Answer Readiness", aeo.get("answer_readiness", 0)],
        ["AEO — Citation Readiness", aeo.get("citation_readiness", 0)],
        ["Overall", scores.get("overall_score", 0)],
    ]
    table = Table(data, colWidths=[300, 80])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ]
        )
    )
    elems.append(Paragraph("Scores", styles["Heading2"]))
    elems.append(table)
    elems.append(PageBreak())

    # Actions
    elems.append(Paragraph("Prioritized Actions", styles["Heading2"]))
    for act in synth.get("prioritized_actions", []):
        elems.append(Paragraph(f"- {act['title']} ({act['impact']}/{act['effort']})", styles["Heading3"]))
        elems.append(Paragraph(act.get("why_it_matters", ""), styles["Normal"]))
        ac = act.get("acceptance_criteria") or []
        if ac:
            elems.append(Paragraph("Acceptance Criteria:", styles["Italic"]))
            for c in ac:
                elems.append(Paragraph(f"• {c}", styles["Normal"]))
        elems.append(Spacer(1, 8))
    elems.append(PageBreak())

    # Risks
    risks = synth.get("key_risks") or []
    if risks:
        elems.append(Paragraph("Key Risks", styles["Heading2"]))
        for r in risks:
            elems.append(Paragraph(f"- {r}", styles["Normal"]))
        elems.append(PageBreak())

    # Crawl quick wins (appendix)
    if crawl:
        elems.append(Paragraph("Crawl Quick Wins", styles["Heading2"]))
        for win in crawl.get("quick_wins", []):
            elems.append(Paragraph(f"- {win['type']}", styles["Normal"]))
        elems.append(PageBreak())

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
                "synthesis": bool(synth),
            },
        },
    }
