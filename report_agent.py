import os, json, base64, io
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from psycopg.rows import dict_row
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors

from openai import OpenAI
from rag_helper import get_rag_context

OPENAI_MODEL = os.getenv("REPORT_LLM_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "30"))
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
# Local analytics on crawl output (hard, per URL)
# -----------------------------------------------------
def _page_issue_flags(p: dict) -> List[str]:
    issues = list(p.get("issues") or [])
    if not p.get("h1"): issues.append("missing_h1")
    if not p.get("meta_description"): issues.append("missing_meta_description")
    canon = p.get("canonical")
    final = p.get("final_url") or p.get("url") or ""
    if canon and isinstance(canon, str):
        if canon.strip("/").lower() != final.strip("/").lower():
            issues.append("canonical_differs")
    if p.get("noindex"): issues.append("noindex_present")
    return sorted(set(issues))

def _summarize_crawl(crawl: Optional[dict]) -> Dict[str, Any]:
    if not crawl: 
        return {"summary": {}, "top_issues": [], "per_page": []}
    pages = crawl.get("pages") or []
    per_page = []
    issue_count = {}
    for p in pages:
        url = p.get("final_url") or p.get("url")
        flags = _page_issue_flags(p)
        for f in flags: issue_count[f] = issue_count.get(f,0) + 1
        per_page.append({
            "url": url,
            "status": p.get("status"),
            "title": p.get("title"),
            "h1": p.get("h1"),
            "meta": p.get("meta_description"),
            "canonical": p.get("canonical"),
            "noindex": bool(p.get("noindex")),
            "issues": flags[:]
        })
    top_issues = sorted(issue_count.items(), key=lambda x: x[1], reverse=True)
    return {
        "summary": crawl.get("summary") or {},
        "top_issues": top_issues,
        "per_page": per_page
    }

def _schema_coverage(schema_out: Optional[dict]) -> Dict[str, Any]:
    if not schema_out: return {"types": [], "faq_pairs": 0}
    sc = schema_out.get("schema")
    types = []
    faq_pairs = 0
    try:
        if isinstance(sc, dict):
            t = sc.get("@type")
            if t: types.append(t if isinstance(t, str) else str(t))
            if sc.get("mainEntity") and isinstance(sc["mainEntity"], list):
                for q in sc["mainEntity"]:
                    if isinstance(q, dict) and q.get("@type") == "Question" and isinstance(q.get("acceptedAnswer"), dict):
                        faq_pairs += 1
    except Exception:
        pass
    return {"types": sorted(set(types)), "faq_pairs": faq_pairs}

def _faq_quality(faq_out: Optional[dict]) -> Dict[str, Any]:
    if not faq_out: return {"count": 0, "avg_len": 0, "with_source": 0}
    faqs = faq_out.get("faqs") or []
    if not faqs: return {"count": 0, "avg_len": 0, "with_source": 0}
    total_words = 0
    with_src = 0
    for f in faqs:
        a = (f.get("a") or "").strip()
        total_words += len(a.split())
        if f.get("source"): with_src += 1
    avg_len = round(total_words / max(1, len(faqs)))
    return {"count": len(faqs), "avg_len": avg_len, "with_source": with_src}

def _keyword_outline(kws: Optional[dict]) -> Dict[str, Any]:
    if not kws: 
        return {"n": 0, "clusters": {}, "suggestions": []}
    return {
        "n": len(kws.get("keywords") or []),
        "clusters": kws.get("clusters") or {},
        "suggestions": kws.get("suggestions") or []
    }

# -----------------------------------------------------
# LLM synthesis (NU: scherper, per-pagina en plan-tier aware)
# -----------------------------------------------------
def _safe(obj, maxlen=4500):
    try:
        s = json.dumps(obj) if not isinstance(obj, str) else obj
        return s[:maxlen]
    except Exception:
        return str(obj)[:maxlen]

def _llm_synthesis(site_meta, crawl_synth, schema_cov, faq_q, kw, rag_kb_text):
    sys = (
        "You are a senior SEO/GEO/AEO auditor. Be blunt and specific. "
        "Point to exact URLs. For each pillar give concrete fixes with acceptance criteria. "
        "Map actions to tiers: Basic, Standard, Premium."
    )
    user = f"""
SITE: {site_meta.get('account_name') or 'Site'} • {site_meta.get('url') or ''} ({site_meta.get('language','')}-{site_meta.get('country','')})

CRAWL (per_page + top_issues):
{_safe(crawl_synth, 4500)}

SCHEMA COVERAGE:
{_safe(schema_cov, 800)}

FAQ QUALITY:
{_safe(faq_q, 400)}

KEYWORDS SNAPSHOT:
{_safe(kw, 1200)}

KB CONTEXT (best practices & policies):
{rag_kb_text[:1800] if rag_kb_text else ''}

Return STRICT JSON:
{{
  "summary": {{
    "title": "SEO • GEO • AEO Audit — {site_meta.get('account_name','Site')}",
    "executive": "2–4 short paragraphs: where we are, biggest blockers, what to do first.",
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
      "pillar": "seo|geo|aeo",
      "tier": "basic|standard|premium",
      "urls": ["https://...", "..."],
      "why_it_matters": "one-liner",
      "impact": "high|medium|low",
      "effort": "low|medium|high",
      "acceptance_criteria": ["measurable bullet 1", "measurable bullet 2"]
    }}
  ],
  "issue_breakdown": {{
    "top_issues": [["issue","count"], ...],
    "worst_pages": [{{"url":"...","issues":["..."]}}]
  }},
  "roadmap_quarterly": {{
    "q1": ["..."],
    "q2": ["..."],
    "q3": ["..."],
    "q4": ["..."]
  }}
}}
Rules:
- Base everything on inputs; if unknown, omit.
- 'urls' must contain the worst pages for that action when relevant.
- Acceptance criteria must be testable (e.g., 'meta description present (<=160 chars) on /gseo').
- Keep JSON tight; no prose outside fields.
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
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {
            "summary": {
                "title": "Audit",
                "executive": "LLM synthesis failed",
                "scores": {"seo": {}, "geo": {}, "aeo": {}, "overall_score": 0},
            },
            "prioritized_actions": [],
            "issue_breakdown": {"top_issues": [], "worst_pages": []},
            "roadmap_quarterly": {"q1":[],"q2":[],"q3":[],"q4":[]}
        }

# -----------------------------------------------------
# PDF helpers
# -----------------------------------------------------
def _styles():
    styles = getSampleStyleSheet()
    # voorkom "Style 'Code' already defined"
    if "CodeMono" not in styles:
        styles.add(ParagraphStyle(name="CodeMono", fontName="Courier", fontSize=8, leading=10))
    return styles

def _table(data: List[List[Any]], col_widths=None):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ALIGN", (0,0), (-1,0), "CENTER"),
    ]))
    return t

def _bullet_list(items: List[str], styles):
    out = []
    for it in items or []:
        out.append(Paragraph(f"• {it}", styles["Normal"]))
    return out

# -----------------------------------------------------
# Report generation
# -----------------------------------------------------
def generate_report(conn, job):
    site_id = job["site_id"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    crawl = _fetch_latest_job(conn, site_id, "crawl")
    faq = _fetch_latest_job(conn, site_id, "faq")
    schema = _fetch_latest_job(conn, site_id, "schema")
    keywords = _fetch_latest_job(conn, site_id, "keywords")
    site_meta = _fetch_site_meta(conn, site_id)

    crawl_synth = _summarize_crawl(crawl)
    schema_cov = _schema_coverage(schema)
    faq_q = _faq_quality(faq)
    kw = _keyword_outline(keywords)

    rag = get_rag_context(conn, site_id=site_id, query="site audit baseline", kb_tags=["SEO","Schema","Quality"])
    rag_kb_text = rag.get("kb_ctx") or ""

    synth = _llm_synthesis(site_meta, crawl_synth, schema_cov, faq_q, kw, rag_kb_text)

    # PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = _styles()
    elems: List[Any] = []

    # Cover
    elems.append(Paragraph(synth["summary"]["title"], styles["Title"]))
    elems.append(Paragraph(f"Generated: {now}", styles["Normal"]))
    elems.append(Paragraph(f"Site: {site_meta.get('account_name','Site')} — {site_meta.get('url','')}", styles["Normal"]))
    elems.append(Spacer(1, 14))
    elems.append(Paragraph(synth["summary"]["executive"], styles["Normal"]))
    elems.append(PageBreak())

    # Scores
    scores = synth.get("summary", {}).get("scores", {})
    data_scores = [
        ["Pillar / Metric", "Score (1–10)"],
        ["SEO — Technical Health", scores.get("seo", {}).get("technical_health", 0)],
        ["SEO — Content Quality", scores.get("seo", {}).get("content_quality", 0)],
        ["SEO — Crawl & Indexability", scores.get("seo", {}).get("crawl_indexability", 0)],
        ["GEO — Entity/Schema Coverage", scores.get("geo", {}).get("entity_schema_coverage", 0)],
        ["GEO — E-E-A-T / Authority", scores.get("geo", {}).get("eeat_authority", 0)],
        ["AEO — Answer Readiness", scores.get("aeo", {}).get("answer_readiness", 0)],
        ["AEO — Citation Readiness", scores.get("aeo", {}).get("citation_readiness", 0)],
        ["Overall", scores.get("overall_score", 0)],
    ]
    elems.append(Paragraph("Scores & Benchmarks", styles["Heading2"]))
    elems.append(_table(data_scores, [320, 80]))
    elems.append(PageBreak())

    # Issue breakdown (from crawl)
    elems.append(Paragraph("Top Issues (site-wide)", styles["Heading2"]))
    top_issues_rows = [["Issue", "# Pages"]]
    for it, cnt in crawl_synth.get("top_issues", []):
        top_issues_rows.append([it, cnt])
    elems.append(_table(top_issues_rows, [320, 80]))
    elems.append(Spacer(1, 12))

    elems.append(Paragraph("Worst Pages (fix first)", styles["Heading2"]))
    worst_rows = [["URL", "Issues"]]
    # pick top 10 by number of issues
    per_page_sorted = sorted(crawl_synth.get("per_page", []), key=lambda x: len(x.get("issues",[])), reverse=True)[:10]
    for p in per_page_sorted:
        worst_rows.append([p.get("url",""), ", ".join(p.get("issues",[])) or "-"])
    elems.append(_table(worst_rows, [350, 150]))
    elems.append(PageBreak())

    # Schema / FAQ / Keywords snapshots
    elems.append(Paragraph("GEO & AEO Coverage Snapshot", styles["Heading2"]))
    elems.append(Paragraph(f"Schema types present (from latest generation): {', '.join(schema_cov.get('types') or ['n/a'])}", styles["Normal"]))
    elems.append(Paragraph(f"FAQ pairs available: {schema_cov.get('faq_pairs',0)}  •  FAQ quality: {faq_q.get('count',0)} items, avg answer length ≈ {faq_q.get('avg_len',0)} words, with source: {faq_q.get('with_source',0)}", styles["Normal"]))
    elems.append(Spacer(1, 10))

    elems.append(Paragraph("Keyword Clusters (snapshot)", styles["Heading3"]))
    clusters = kw.get("clusters") or {}
    data_kw = [["Cluster", "Examples"]]
    for k,v in (clusters.items() if isinstance(clusters, dict) else []):
        data_kw.append([k, ", ".join(v[:6])])
    elems.append(_table(data_kw, [200, 300]))
    elems.append(Spacer(1, 10))

    suggs = kw.get("suggestions") or []
    if suggs:
        elems.append(Paragraph("Suggested Pages (from context)", styles["Heading3"]))
        for s in suggs[:6]:
            elems.append(Paragraph(f"• {s.get('page_title','')} — {', '.join(s.get('grouped_keywords',[])[:5])}", styles["Normal"]))
    elems.append(PageBreak())

    # Prioritized actions
    elems.append(Paragraph("Prioritized Actions (with Tiers)", styles["Heading2"]))
    actions = synth.get("prioritized_actions") or []
    for a in actions:
        title = a.get("title","")
        pillar = a.get("pillar","")
        tier = a.get("tier","")
        impact = a.get("impact","")
        effort = a.get("effort","")
        urls = a.get("urls") or []
        elems.append(Paragraph(f"{title}  —  [{pillar.upper()} • {tier.upper()}]  ({impact}/{effort})", styles["Heading3"]))
        if urls:
            elems.append(Paragraph("Pages:", styles["Italic"]))
            for u in urls[:10]:
                elems.append(Paragraph(f"• {u}", styles["Normal"]))
        ac = a.get("acceptance_criteria") or []
        if ac:
            elems.append(Paragraph("Acceptance criteria:", styles["Italic"]))
            elems.extend(_bullet_list(ac, styles))
        why = a.get("why_it_matters","")
        if why:
            elems.append(Paragraph(f"Why it matters: {why}", styles["Normal"]))
        elems.append(Spacer(1, 8))
    elems.append(PageBreak())

    # Quarterly roadmap (Premium expectation)
    roadmap = synth.get("roadmap_quarterly") or {}
    elems.append(Paragraph("Quarterly Roadmap (Premium)", styles["Heading2"]))
    for q in ["q1","q2","q3","q4"]:
        items = roadmap.get(q) or []
        elems.append(Paragraph(q.upper(), styles["Heading3"]))
        elems.extend(_bullet_list(items, styles))
        elems.append(Spacer(1, 6))

    # Build
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
                "synthesis": True
            },
        },
    }
