#!/usr/bin/env python3
import os
import io
import json
import base64
from datetime import datetime, timezone
from typing import Any, Dict, List

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER

API_BASE = os.getenv("API_BASE_URL", "https://aseon-api.onrender.com")

REPORT_TITLE = "Monthly SEO/AEO/GEO Report"
REPORT_SECTIONS_DEFAULT = ["crawl", "keywords", "faq", "schema", "quick_wins", "plan"]
REPORT_FORMAT_DEFAULT = "markdown"  # markdown | html | pdf | both


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


def _latest_outputs_by_type(conn, site_id: str, types: List[str]) -> Dict[str, Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (type) id, type, output, finished_at
              FROM jobs
             WHERE site_id=%s
               AND status='done'
               AND type = ANY(%s)
             ORDER BY type, finished_at DESC NULLS LAST, created_at DESC
            """,
            (site_id, types),
        )
        out = {}
        for row in cur.fetchall():
            out[row["type"]] = {
                "id": row["id"],
                "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
                "output": row["output"] or {},
            }
        return out


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
        lines.append(f"- Pages crawled: {c.get('pages_crawled', '?')}")
        issues = c.get("issues") or []
        if issues:
            lines.append("**Issues:**")
            for i in issues[:10]:
                lines.append(f"- {i}")
        lines.append("")

    if "keywords" in include_sections and "keywords" in latest:
        k = latest["keywords"]["output"] or {}
        clusters = k.get("clusters") or {}
        lines.append("## Keyword clusters")
        for intent, terms in clusters.items():
            lines.append(f"- **{intent}**: {', '.join(terms[:5])}…")
        lines.append("")

    if "faq" in include_sections and "faq" in latest:
        f = latest["faq"]["output"] or {}
        faqs = f.get("faqs") or []
        lines.append("## FAQ’s")
        for qa in faqs[:5]:
            lines.append(f"- **Q:** {qa.get('q')}")
            lines.append(f"  - A: {qa.get('a')}")
        lines.append("")

    if "schema" in include_sections and "schema" in latest:
        s = latest["schema"]["output"] or {}
        schema = s.get("schema")
        lines.append("## Schema JSON-LD (preview)")
        if schema:
            lines.append("```json")
            lines.append(json.dumps(schema, indent=2))
            lines.append("```")
        lines.append("")

    if "quick_wins" in include_sections:
        qws = []
        if "crawl" in latest:
            qws += _safe_get(latest["crawl"]["output"], ["quick_wins"], []) or []
        if qws:
            lines.append("## Quick wins")
            for q in qws:
                lines.append(f"- {q}")
        lines.append("")

    if "plan" in include_sections:
        lines.append("## Monthly plan")
        plan = ["Fix crawl issues", "Publish FAQ block", "Validate schema", "Add content for keyword clusters"]
        for p in plan:
            lines.append(f"- {p}")

    return "\n".join(lines)


# ---------- PDF rendering ----------

def _render_pdf(title: str, site: Dict[str, Any], latest: Dict[str, Any], include_sections: List[str]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=title)
    styles = getSampleStyleSheet()
    story = []

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 24))
    story.append(Paragraph(f"Site: {site.get('url')} ({site.get('language')}/{site.get('country')})", styles["Normal"]))
    story.append(Paragraph(f"Generated at {_now()}", styles["Normal"]))
    story.append(Spacer(1, 36))

    def add_section(heading: str, items: List[str]):
        story.append(Paragraph(heading, styles["Heading2"]))
        story.append(Spacer(1, 6))
        if items:
            lf = ListFlowable([ListItem(Paragraph(i, styles["Normal"])) for i in items], bulletType="bullet")
            story.append(lf)
        story.append(Spacer(1, 12))

    if "crawl" in include_sections and "crawl" in latest:
        c = latest["crawl"]["output"] or {}
        issues = [str(i) for i in (c.get("issues") or []) if i][:10]
        add_section("Crawl summary", [f"Pages crawled: {c.get('pages_crawled','?')}"] + issues)

    if "keywords" in include_sections and "keywords" in latest:
        k = latest["keywords"]["output"] or {}
        clusters = k.get("clusters") or {}
        kw_items = [f"{intent}: {', '.join(terms[:5])}…" for intent, terms in clusters.items()]
        add_section("Keyword clusters", kw_items)

    if "faq" in include_sections and "faq" in latest:
        f = latest["faq"]["output"] or {}
        faqs = f.get("faqs") or []
        faq_items = [f"Q: {qa.get('q')} — A: {qa.get('a')}" for qa in faqs[:5]]
        add_section("FAQs", faq_items)

    if "schema" in include_sections and "schema" in latest:
        add_section("Schema (JSON-LD)", ["see API output for full JSON"])

    if "quick_wins" in include_sections:
        qws = []
        if "crawl" in latest:
            qws += _safe_get(latest["crawl"]["output"], ["quick_wins"], []) or []
        add_section("Quick wins", qws)

    if "plan" in include_sections:
        plan = ["Fix crawl issues", "Publish FAQ block", "Validate schema", "Add content for keyword clusters"]
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
    site = {"url": "?", "language": "?", "country": "?"}
    with conn.cursor() as cur:
        cur.execute("SELECT url, language, country FROM sites WHERE id=%s", (site_id,))
        row = cur.fetchone()
        if row:
            site = dict(row)

    out: Dict[str, Any] = {
        "site": site,
        "title": title,
        "generated_at": _now(),
        "format": fmt,
    }

    # Always add markdown
    md = _render_markdown(title, site, latest, include_sections)
    out["markdown"] = md

    if fmt in ("html", "both"):
        out["html"] = f"<pre>{md}</pre>"

    if fmt in ("pdf", "both"):
        pdf_bytes = _render_pdf(title, site, latest, include_sections)
        out["pdf_base64"] = base64.b64encode(pdf_bytes).decode("utf-8")

    return out
