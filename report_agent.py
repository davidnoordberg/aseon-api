# report_agent.py
import os
import io
import json
import base64
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from psycopg.rows import dict_row
from openai import OpenAI

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
    KeepTogether,
)

from rag_helper import get_rag_context

# --------- LLM Setup ----------
OPENAI_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "30"))
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# --------- DB Helpers ----------
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

# --------- Utils ----------
def _safe(obj: Any, maxlen: int = 4000) -> str:
    try:
        s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
        return s[:maxlen]
    except Exception:
        return str(obj)[:maxlen]

def _json_or_default(text: str, default: Any) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return default

def _avg(nums: List[float]) -> float:
    nums = [float(n) for n in nums if isinstance(n, (int, float))]
    return round(sum(nums) / len(nums), 1) if nums else 0.0

def _cap(s: Optional[str], n: int = 700) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"

# --------- RAG Contexts per specialist ----------
def _rag_for_seo(conn, site_id: str) -> Dict[str, Any]:
    # Techniek + content + crawl/indexability
    return get_rag_context(
        conn,
        site_id=site_id,
        query="technical SEO crawl indexability content quality internal links duplicates canonicals metadata",
        kb_tags=["SEO", "Technical", "Content"]
    )

def _rag_for_geo(conn, site_id: str) -> Dict[str, Any]:
    # Entities + schema + E-E-A-T
    return get_rag_context(
        conn,
        site_id=site_id,
        query="structured data schema.org organization person website sameAs article breadcrumb eeat experience expertise authoritativeness trust",
        kb_tags=["Schema", "Entities", "Quality", "E-E-A-T"]
    )

def _rag_for_aeo(conn, site_id: str) -> Dict[str, Any]:
    # Answer & citation readiness
    return get_rag_context(
        conn,
        site_id=site_id,
        query="answer engine optimization FAQ question answering snippets quotable sections citations sources",
        kb_tags=["AEO", "FAQ", "Content", "Schema", "Quality"]
    )

# --------- Prompts (met strikte JSON schema’s) ----------

SEO_PROMPT = """
You are a senior **SEO specialist**. Use ONLY the provided inputs + RAG context. Cite inline with [S#] for site docs and [K#] for KB where relevant.

Site: {site_url}
Inputs (summaries):
- Crawl: {crawl}
- Keywords: {keywords}

RAG Context (site & KB, trimmed):
--- SITE ---
{site_ctx}
--- KB ---
{kb_ctx}

Return STRICT JSON (no prose, no markdown):

{{
  "scores": {{
    "technical_health": 0-10,
    "content_quality": 0-10,
    "crawl_indexability": 0-10
  }},
  "findings": [
    {{
      "page_url": "https://...",
      "issue": "short title / missing meta description / canonical mismatch / thin content / ...",
      "severity": "high|medium|low",
      "why": "1-2 sentences, factual",
      "fix": "precise, actionable steps",
      "acceptance_criteria": ["measurable check 1","measurable check 2"],
      "evidence": "short quote/snippet from context if available",
      "source_ids": ["S1","K2"]
    }}
  ],
  "top_wins": [
    {{
      "title": "Fix X across N pages",
      "impact": "high|medium|low",
      "effort": "low|medium|high",
      "owner": "dev|content|seo"
    }}
  ]
}}
Rules:
- Only include pages you can back up with site context (S#). If you generalize, note it.
- Keep issues concrete and testable.
- Do not invent URLs. If unsure, omit the item.
"""

GEO_PROMPT = """
You are a **GEO / Entity SEO specialist**. Focus structured data coverage, entity disambiguation, and E-E-A-T signals. Use ONLY inputs + RAG. Cite with [S#]/[K#].

Site: {site_url}
Inputs:
- Schema output: {schema}
- Crawl (headings/meta snippets available): {crawl}

RAG Context:
--- SITE ---
{site_ctx}
--- KB ---
{kb_ctx}

Return STRICT JSON:

{{
  "scores": {{
    "entity_schema_coverage": 0-10,
    "eeat_authority": 0-10
  }},
  "findings": [
    {{
      "page_url": "https://...",
      "entity": "Organization|Person|Article|WebSite|BreadcrumbList|FAQPage|LocalBusiness|...",
      "gap": "missing required property X / stale logo URL / no sameAs / no BreadcrumbList / ...",
      "severity": "high|medium|low",
      "fix": "exact JSON-LD property additions/changes",
      "acceptance_criteria": ["RR test passes for type X","property Y present with value Z"],
      "source_ids": ["S2","K1"]
    }}
  ],
  "top_wins": [
    {{
      "title": "Implement Organization + Logo + sameAs",
      "impact": "high|medium|low",
      "effort": "low|medium|high",
      "owner": "dev|seo"
    }}
  ]
}}
Rules:
- Prefer Organization, WebSite, Article, BreadcrumbList, Person where applicable.
- Only claim what is present/missing in context.
- Keep fixes JSON-LD-ready (property names as in schema.org).
"""

AEO_PROMPT = """
You are an **AEO specialist** (Answer Engine Optimization). Optimize for answer-readiness and citation-readiness. Use ONLY inputs + RAG. Cite with [S#]/[K#].

Site: {site_url}
Inputs:
- FAQ: {faq}
- Schema: {schema}
- Crawl: {crawl}

RAG Context:
--- SITE ---
{site_ctx}
--- KB ---
{kb_ctx}

Return STRICT JSON:

{{
  "scores": {{
    "answer_readiness": 0-10,
    "citation_readiness": 0-10
  }},
  "findings": [
    {{
      "page_url": "https://...",
      "question": "the user-intent question",
      "gap": "no concise answer / lacks sources / no anchor / ...",
      "severity": "high|medium|low",
      "fix": "create/upgrade an answer block (≤80 words), add anchor, link to canonical page",
      "acceptance_criteria": ["answer block present ≤80 words","anchor #faq-x exists","internal link from hub page"],
      "source_ids": ["S1","K3"]
    }}
  ],
  "top_wins": [
    {{
      "title": "Create answer fragments for top 5 intents",
      "impact": "high|medium|low",
      "effort": "low|medium|high",
      "owner": "content|seo"
    }}
  ]
}}
Rules:
- Answers must be quotable: ≤80 words, self-contained, fact-based.
- If FAQ/schema mismatch visible content, flag it.
- Only return pages present in site context.
"""

EXEC_SUMMARY_PROMPT = """
You are a lead auditor. Create a sharp executive summary (2–4 short paragraphs) based on the merged SEO/GEO/AEO results below. No marketing fluff; be specific.

Inputs:
- Site: {site_url} ({lang}-{country})
- Scores: {scores}
- SEO findings (top 10): {seo_findings}
- GEO findings (top 10): {geo_findings}
- AEO findings (top 10): {aeo_findings}

Return PLAIN TEXT, max ~1400 chars, no markdown.
"""

# --------- LLM helpers ----------
def _call_specialist(prompt_text: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        timeout=OPENAI_TIMEOUT_SEC,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a precise auditor. Output strict JSON only."},
            {"role": "user", "content": prompt_text},
        ],
    )
    return _json_or_default(resp.choices[0].message.content, {})

def _call_summary(prompt_text: str) -> str:
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        timeout=OPENAI_TIMEOUT_SEC,
        messages=[
            {"role": "system", "content": "You are concise and concrete."},
            {"role": "user", "content": prompt_text},
        ],
    )
    return (resp.choices[0].message.content or "").strip()

# --------- Synthesis / scoring ----------
def _compute_overall(seo_scores: Dict[str, float], geo_scores: Dict[str, float], aeo_scores: Dict[str, float]) -> float:
    parts = []
    for k in ("technical_health","content_quality","crawl_indexability"):
        if k in seo_scores: parts.append(seo_scores[k])
    for k in ("entity_schema_coverage","eeat_authority"):
        if k in geo_scores: parts.append(geo_scores[k])
    for k in ("answer_readiness","citation_readiness"):
        if k in aeo_scores: parts.append(aeo_scores[k])
    return _avg(parts)

def _top_n(lst: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    return (lst or [])[:n]

def _priority_actions(seo: Dict[str, Any], geo: Dict[str, Any], aeo: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Gebruik top_wins uit elk domein; sorteer impact/effort heuristisch
    def weight(win):
        imp = {"high": 3, "medium": 2, "low": 1}.get((win.get("impact") or "").lower(), 1)
        eff = {"low": 3, "medium": 2, "high": 1}.get((win.get("effort") or "").lower(), 2)
        return -(imp*10 + eff)  # hoger impact + lager effort eerst
    wins = (seo.get("top_wins") or []) + (geo.get("top_wins") or []) + (aeo.get("top_wins") or [])
    wins = sorted(wins, key=weight)
    return wins[:10]

# --------- PDF helpers ----------
def _styles():
    styles = getSampleStyleSheet()
    # voorkom naamconflict met "Code"
    if "MonoSmall" not in styles:
        styles.add(ParagraphStyle(name="MonoSmall", fontName="Courier", fontSize=8, leading=9))
    return styles

def _table(data: List[List[Any]], col_widths: Optional[List[int]] = None) -> Table:
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
    ]))
    return t

def _finding_rows(findings: List[Dict[str, Any]], cols: List[str]) -> List[List[str]]:
    rows = []
    for f in findings:
        row = []
        for c in cols:
            v = f.get(c)
            if isinstance(v, list):
                v = " • ".join([str(x) for x in v])
            row.append(_cap(str(v or "")))
        rows.append(row)
    return rows

# --------- Main entry ----------
def generate_report(conn, job):
    site_id = job["site_id"]
    payload = job.get("payload", {}) or {}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Inputs
    crawl = _fetch_latest_job(conn, site_id, "crawl")
    faq = _fetch_latest_job(conn, site_id, "faq")
    schema = _fetch_latest_job(conn, site_id, "schema")
    keywords = _fetch_latest_job(conn, site_id, "keywords")
    site_meta = _fetch_site_meta(conn, site_id)
    site_url = (site_meta.get("url") or "").strip()
    lang = (site_meta.get("language") or "").strip()
    country = (site_meta.get("country") or "").strip()

    # RAG per specialist
    rag_seo = _rag_for_seo(conn, site_id)
    rag_geo = _rag_for_geo(conn, site_id)
    rag_aeo = _rag_for_aeo(conn, site_id)

    # Specialisten aanroepen
    seo_prompt = SEO_PROMPT.format(
        site_url=site_url,
        crawl=_safe(crawl, 1800),
        keywords=_safe(keywords, 1200),
        site_ctx=_safe(rag_seo.get("site_ctx"), 2200),
        kb_ctx=_safe(rag_seo.get("kb_ctx"), 1600),
    )
    geo_prompt = GEO_PROMPT.format(
        site_url=site_url,
        schema=_safe(schema, 1600),
        crawl=_safe(crawl, 1200),
        site_ctx=_safe(rag_geo.get("site_ctx"), 2200),
        kb_ctx=_safe(rag_geo.get("kb_ctx"), 1600),
    )
    aeo_prompt = AEO_PROMPT.format(
        site_url=site_url,
        faq=_safe(faq, 1200),
        schema=_safe(schema, 800),
        crawl=_safe(crawl, 800),
        site_ctx=_safe(rag_aeo.get("site_ctx"), 2200),
        kb_ctx=_safe(rag_aeo.get("kb_ctx"), 1600),
    )

    seo = _call_specialist(seo_prompt) or {}
    geo = _call_specialist(geo_prompt) or {}
    aeo = _call_specialist(aeo_prompt) or {}

    # Scores + overall
    seo_scores = seo.get("scores") or {}
    geo_scores = geo.get("scores") or {}
    aeo_scores = aeo.get("scores") or {}
    overall = _compute_overall(seo_scores, geo_scores, aeo_scores)

    # Executive summary
    exec_text = _call_summary(EXEC_SUMMARY_PROMPT.format(
        site_url=site_url, lang=lang, country=country,
        scores=json.dumps({"seo": seo_scores, "geo": geo_scores, "aeo": aeo_scores, "overall": overall}),
        seo_findings=json.dumps(_top_n(seo.get("findings") or [], 10), ensure_ascii=False),
        geo_findings=json.dumps(_top_n(geo.get("findings") or [], 10), ensure_ascii=False),
        aeo_findings=json.dumps(_top_n(aeo.get("findings") or [], 10), ensure_ascii=False),
    ))

    # Prioritized actions
    actions = _priority_actions(seo, geo, aeo)

    # Build PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = _styles()
    elems: List[Any] = []

    # Title + executive
    title = f"SEO • GEO • AEO Audit — {site_url or 'Site'}"
    elems.append(Paragraph(title, styles["Title"]))
    elems.append(Paragraph(f"Generated: {now}", styles["Normal"]))
    elems.append(Spacer(1, 10))
    if exec_text:
        elems.append(Paragraph(exec_text, styles["Normal"]))
    elems.append(PageBreak())

    # Score table
    score_data = [
        ["Pillar / Metric", "Score (1–10)"],
        ["SEO — Technical Health", seo_scores.get("technical_health", 0)],
        ["SEO — Content Quality", seo_scores.get("content_quality", 0)],
        ["SEO — Crawl & Indexability", seo_scores.get("crawl_indexability", 0)],
        ["GEO — Entity/Schema Coverage", geo_scores.get("entity_schema_coverage", 0)],
        ["GEO — E-E-A-T / Authority", geo_scores.get("eeat_authority", 0)],
        ["AEO — Answer Readiness", aeo_scores.get("answer_readiness", 0)],
        ["AEO — Citation Readiness", aeo_scores.get("citation_readiness", 0)],
        ["Overall", overall],
    ]
    elems.append(Paragraph("Scores", styles["Heading2"]))
    elems.append(_table(score_data, col_widths=[320, 80]))
    elems.append(PageBreak())

    # Prioritized Actions
    if actions:
        elems.append(Paragraph("Prioritized Actions (Top 10)", styles["Heading2"]))
        act_rows = [["Action", "Impact", "Effort", "Owner"]]
        for a in actions:
            act_rows.append([
                _cap(a.get("title")),
                (a.get("impact") or "").title(),
                (a.get("effort") or "").title(),
                (a.get("owner") or "").upper()
            ])
        elems.append(_table(act_rows, col_widths=[320, 80, 80, 80]))
        elems.append(PageBreak())

    # Findings per domain
    def add_findings(section_title: str, findings: List[Dict[str, Any]], cols: List[str], widths: List[int]):
        elems.append(Paragraph(section_title, styles["Heading2"]))
        if not findings:
            elems.append(Paragraph("No findings.", styles["Normal"]))
            elems.append(Spacer(1, 6))
            return
        header = [c.replace("_", " ").title() for c in cols]
        rows = _finding_rows(findings, cols)
        elems.append(_table([header] + rows[:60], col_widths=widths))
        elems.append(PageBreak())

    add_findings(
        "SEO Findings",
        seo.get("findings") or [],
        cols=["page_url","issue","severity","why","fix","acceptance_criteria","source_ids"],
        widths=[140,110,50,120,120,120,80],
    )
    add_findings(
        "GEO Findings",
        geo.get("findings") or [],
        cols=["page_url","entity","gap","severity","fix","acceptance_criteria","source_ids"],
        widths=[140,80,110,50,140,120,80],
    )
    add_findings(
        "AEO Findings",
        aeo.get("findings") or [],
        cols=["page_url","question","gap","severity","fix","acceptance_criteria","source_ids"],
        widths=[140,120,100,50,140,120,80],
    )

    # Appendix: Raw summaries (compact)
    elems.append(Paragraph("Appendix — Raw Inputs (Truncated)", styles["Heading2"]))
    elems.append(Paragraph("Crawl", styles["Heading3"]))
    elems.append(Paragraph(_cap(_safe(crawl, 3000), 3000), styles["MonoSmall"]))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph("Keywords", styles["Heading3"]))
    elems.append(Paragraph(_cap(_safe(keywords, 3000), 3000), styles["MonoSmall"]))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph("FAQ", styles["Heading3"]))
    elems.append(Paragraph(_cap(_safe(faq, 3000), 3000), styles["MonoSmall"]))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph("Schema", styles["Heading3"]))
    elems.append(Paragraph(_cap(_safe(schema, 3000), 3000), styles["MonoSmall"]))
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
                "seo": bool(seo),
                "geo": bool(geo),
                "aeo": bool(aeo),
                "actions": bool(actions),
            },
        },
    }
