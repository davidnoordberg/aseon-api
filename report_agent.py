import os
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

OPENAI_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "30"))
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

def _safe(obj, maxlen=4000):
    try:
        s = json.dumps(obj) if not isinstance(obj, str) else obj
        return s[:maxlen]
    except Exception:
        return str(obj)[:maxlen]

def _llm_synthesis(site_meta, crawl, keywords, faq, schema, rag):
    sys = (
        "You are a senior SEO/GEO/AEO auditor. Be concrete, professional, and actionable. "
        "Ground advice in inputs/KB only; no hallucinations. Use crisp language."
    )
    user = f"""
SITE:
- name: {site_meta.get('account_name') or 'Site'}
- url: {site_meta.get('url') or ''} ({site_meta.get('language') or ''}-{site_meta.get('country') or ''})

INPUTS (summaries; truncated):
- CRAWL: { _safe(crawl, 3500) if crawl else "null" }
- KEYWORDS: { _safe(keywords, 3500) if keywords else "null" }
- FAQ: { _safe(faq, 3500) if faq else "null" }
- SCHEMA: { _safe(schema, 3500) if schema else "null" }

KB CONTEXT (policies/best practices):
{ rag.get('kb_ctx') or '' }

Return STRICT JSON:
{{
  "summary": {{
    "title": "SEO • GEO • AEO Audit for {site_meta.get('account_name') or 'Site'}",
    "executive": "2–3 tight paragraphs. No fluff.",
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
      "title": "Verb with object (max 8 words)",
      "why_it_matters": "One sentence, measurable rationale",
      "impact": "high|medium|low",
      "effort": "low|medium|high",
      "owner_hint": "content|dev|seo",
      "acceptance_criteria": [
        "Observable metric 1",
        "Observable metric 2"
      ]
    }}
  ],
  "key_risks": ["Short risk 1","Short risk 2"]
}}
Rules:
- Scores are integers 0–10 only.
- Max 8 actions. Prefer high-impact, low/medium-effort first.
- Acceptance criteria must be verifiable (metrics, thresholds, tools).
"""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        timeout=OPENAI_TIMEOUT_SEC,
        response_format={"type": "json_object"},
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
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

def generate_report(conn, job):
    site_id = job["site_id"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    crawl = _fetch_latest_job(conn, site_id, "crawl")
    faq = _fetch_latest_job(conn, site_id, "faq")
    schema = _fetch_latest_job(conn, site_id, "schema")
    keywords = _fetch_latest_job(conn, site_id, "keywords")
    site_meta = _fetch_site_meta(conn, site_id)

    rag = get_rag_context(conn, site_id=site_id, query="site audit baseline")
    synth = _llm_synthesis(site_meta, crawl, keywords, faq, schema, rag)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    if "Code" not in styles.byName:
        styles.add(ParagraphStyle(name="Code", fontName="Courier", fontSize=8))

    elems = []
    elems.append(Paragraph(synth["summary"]["title"], styles["Title"]))
    elems.append(Paragraph(f"Generated at: {now}", styles["Normal"]))
    elems.append(Spacer(1, 12))
    elems.append(Paragraph(synth["summary"]["executive"], styles["Normal"]))
    elems.append(PageBreak())

    scores = synth["summary"]["scores"]
    seo = scores.get("seo", {}) or {}
    geo = scores.get("geo", {}) or {}
    aeo = scores.get("aeo", {}) or {}

    data = [
        ["Pillar / Metric", "Score (0–10)"],
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
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ]))
    elems.append(Paragraph("Scores", styles["Heading2"]))
    elems.append(table)
    elems.append(PageBreak())

    elems.append(Paragraph("Prioritized Actions", styles["Heading2"]))
    for act in (synth.get("prioritized_actions") or [])[:8]:
        elems.append(Paragraph(f"- {act.get('title','') } ({act.get('impact','')}/{act.get('effort','')})", styles["Heading3"]))
        if act.get("why_it_matters"):
            elems.append(Paragraph(act["why_it_matters"], styles["Normal"]))
        ac = act.get("acceptance_criteria") or []
        if ac:
            elems.append(Paragraph("Acceptance Criteria:", styles["Italic"]))
            for c in ac:
                elems.append(Paragraph(f"• {c}", styles["Normal"]))
        elems.append(Spacer(1, 8))
    elems.append(PageBreak())

    risks = synth.get("key_risks") or []
    if risks:
        elems.append(Paragraph("Key Risks", styles["Heading2"]))
        for r in risks:
            elems.append(Paragraph(f"- {r}", styles["Normal"]))
        elems.append(PageBreak())

    if crawl and crawl.get("quick_wins"):
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
