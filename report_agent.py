#!/usr/bin/env python3
import os
import io
import json
import base64
from datetime import datetime, timezone
from typing import Any, Dict, List

from psycopg.rows import dict_row
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER

API_BASE = os.getenv("API_BASE_URL", "https://aseon-api.onrender.com")

REPORT_TITLE = "Monthly SEO/AEO/GEO Report"
REPORT_SECTIONS_DEFAULT = ["crawl", "keywords", "faq", "schema", "quick_wins", "plan"]
REPORT_FORMAT_DEFAULT = "markdown"  # "markdown" | "html" | "pdf" | "both"


# ---------- utils ----------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


# ---------- data access (robust: geen ANY/arrays, maar 1 query per type) ----------

def _get_site(conn, site_id: str) -> Dict[str, Any]:
    site = {"url": "?", "language": "?", "country": "?"}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT url, language, country FROM sites WHERE id=%s", (site_id,))
        row = cur.fetchone()
        if row:
            site = dict(row)
    return site


def _latest_outputs_by_type(conn, site_id: str, types: List[str]) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    with conn.cursor(row_factory=dict_row) as cur:
        for t in types:
            cur.execute(
                """
                SELECT id, type, output, finished_at
                  FROM jobs
                 WHERE site_id=%s
                   AND status='done'
                   AND type=%s
                 ORDER BY finished_at DESC NULLS LAST, created_at DESC
                 LIMIT 1
                """,
                (site_id, t),
            )
            row = cur.fetchone()
            if row:
                results[t] = {
                    "id": row["id"],
                    "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
                    "output": row.get("output") or {},
                }
    return results


# ---------- Markdown rendering ----------

def _render_markdown(title: str, site: Dict[str, Any], latest: Dict[str, Any], include_sections: List[str]) -> str:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Site:** {site.get('url')} ({site.get('language')}/{site.get('country')})")
    lines.append(f"_Generated at {_now()}_")
    lines.append("")

    if "crawl" in include_sections and "crawl" in latest:
        c = latest["crawl"]["output"] or {}
        lines.append("## Crawl summary")
        pages = c.get("pages_crawled") or _safe_get(c, ["stats", "pages"], "?")
        lines.append(f"- Pages crawled: {pages}")
        issues = c.get("issues") or []
        if issues:
            lines.append("**Issues (top 10):**")
            for i in issues[:10]:
                lines.append(f"- {i}")
        qws = _safe_get(c, ["quick_wins"], []) or []
        if qws:
            lines.append("**Quick wins:**")
            for q in qws[:10]:
                lines.append(f"- {q}")
        lines.append("")

    if "keywords" in include_sections and "keywords" in latest:
        k = latest["keywords"]["output"] or {}
        clusters = k.get("clusters") or {}
        if clusters:
            lines.append("## Keyword clusters")
            for intent, terms in clusters.items():
                short = ", ".join(list(terms)[:5])
                lines.append(f"- **{intent}**: {short}…")
            lines.append("")

    if "faq" in include_sections and "faq" in latest:
        f = latest["faq"]["output"] or {}
        faqs = f.get("faqs") or []
        if faqs:
            lines.append("## FAQ’s")
            for qa in faqs[:5]:
                q = str(qa.get("q", "")).strip()
                a = str(qa.get("a", "")).strip()
                lines.append(f"- **Q:** {q}")
                lines.append(f"  - A: {a}")
            lines.append("")

    if "schema" in include_sections and "schema" in latest:
        s = latest["schema"]["output"] or {}
        schema = s.get("schema")
        lines.append("## Schema JSON-LD (preview)")
        if schema:
            lines.append("```json")
            try:
                lines.append(json.dumps(schema, indent=2, ensure_ascii=False))
            except Exception:
                lines.append(json.dumps(schema))
            lines.append("```")
        lines.append("")

    if "quick_wins" in include_sections:
        agg = []
        if "crawl" in latest:
            agg += _safe_get(latest["crawl"]["output"] or {}, ["quick_wins"], []) or []
        if agg:
            lines.append("## Quick wins (consolidated)")
            for q in agg[:15]:
                lines.append(f"- {q}")
            lines.append("")

    if "plan" in include_sections:
        plan = []
        crawl_issues = []
        if "crawl" in latest:
            crawl_issues = _safe_get(latest["crawl"]["output"] or {}, ["issues"], []) or []
        if crawl_issues:
            plan.append("Fix top technical issues from crawl (titles/meta/robots/canonicals).")
        if "keywords" in latest:
            plan.append("Publish 2 pages targeting strongest keyword clusters.")
        if "faq" in latest:
            plan.append("Add FAQ blocks on 2–3 key pages to target snippets/AI answers.")
        if "schema" in latest:
            plan.append("Validate/monitor JSON-LD (FAQ/Organization/Article) in production.")
        if not plan:
            plan.append("Refresh signals: rerun crawl/keywords/faq/schema.")
        lines.append("## Monthly plan")
        for p in plan:
            lines.append(f"- {p}")

    return "\n".join(lines)


# ---------- PDF rendering ----------

def _render_pdf(title: str, site: Dict[str, Any], latest: Dict[str, Any], include_sections: List[str]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=title)
    styles = getSampleStyleSheet()
    story = []

    tstyle = styles["Title"]; tstyle.alignment = TA_CENTER
    story.append(Paragraph(title, tstyle))
    story.append(Spacer(1, 24))
    story.append(Paragraph(f"Site: {site.get('url')} ({site.get('language')}/{site.get('country')})", styles["Normal"]))
    story.append(Paragraph(f"Generated at {_now()}", styles["Normal"]))
    story.append(Spacer(1, 24))

    def add_section(head: str, items: List[str]):
        story.append(Paragraph(head, styles["Heading2"]))
        story.append(Spacer(1, 6))
        if items:
            lf = ListFlowable([ListItem(Paragraph(i, styles["Normal"])) for i in items], bulletType="bullet")
            story.append(lf)
        story.append(Spacer(1, 12))

    # Crawl
    if "crawl" in include_sections and "crawl" in latest:
        c = latest["crawl"]["output"] or {}
        pages = c.get("pages_crawled") or _safe_get(c, ["stats", "pages"], "?")
        items = [f"Pages crawled: {pages}"]
        items += [str(i) for i in (c.get("issues") or [])][:10]
        add_section("Crawl summary", items)

    # Keywords
    if "keywords" in include_sections and "keywords" in latest:
        k = latest["keywords"]["output"] or {}
        clusters = k.get("clusters") or {}
        kw_items = [f"{intent}: {', '.join(list(terms)[:5])}…" for intent, terms in clusters.items()]
        add_section("Keyword clusters", kw_items)

    # FAQs
    if "faq" in include_sections and "faq" in latest:
        f = latest["faq"]["output"] or {}
        faqs = f.get("faqs") or []
        faq_items = [f"Q: {str(qa.get('q','')).strip()} — A: {str(qa.get('a','')).strip()}" for qa in faqs[:5]]
        add_section("FAQs", faq_items)

    # Schema
    if "schema" in include_sections and "schema" in latest:
        add_section("Schema (JSON-LD)", ["See API output for full JSON."])

    # Quick wins
    if "quick_wins" in include_sections:
        agg = []
        if "crawl" in latest:
            agg += _safe_get(latest["crawl"]["output"] or {}, ["quick_wins"], []) or []
        add_section("Quick wins", [str(x) for x in agg[:15]])

    # Plan
    if "plan" in include_sections:
        plan = [
            "Fix crawl issues",
            "Publish FAQ blocks",
            "Validate JSON-LD",
            "Create 2 pages for top clusters",
        ]
        add_section("Monthly plan", plan)

    doc.build(story)
    return buf.getvalue()


# ---------- main entry ----------

def generate_report(conn, job: Dict[str, Any]) -> Dict[str, Any]:
    site_id = job["site_id"]
    payload = job.get("payload") or {}
    fmt = (payload.get("format") or REPORT_FORMAT_DEFAULT).lower()
    include_sections = payload.get("sections") or REPORT_SECTIONS_DEFAULT
    title = payload.get("title") or REPORT_TITLE

    latest = _latest_outputs_by_type(conn, site_id, ["crawl", "keywords", "faq", "schema"])
    site = _get_site(conn, site_id)

    out: Dict[str, Any] = {
        "site": site,
        "title": title,
        "generated_at": _now(),
        "format": fmt,
    }

    # Altijd markdown
    md = _render_markdown(title, site, latest, include_sections)
    out["markdown"] = md

    # Optioneel HTML
    if fmt in ("html", "both"):
        out["html"] = f"<pre>{md}</pre>"

    # Optioneel PDF
    if fmt in ("pdf", "both"):
        pdf_bytes = _render_pdf(title, site, latest, include_sections)
        out["pdf_base64"] = base64.b64encode(pdf_bytes).decode("utf-8")

    return out
