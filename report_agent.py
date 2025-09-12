#!/usr/bin/env python3
import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Dit bestand gebruikt psycopg3-API via de connectie die door general_agent wordt aangeleverd.
# We doen hier GEEN eigen connection management; uitsluitend pure functions.

API_BASE = os.getenv("API_BASE_URL", "https://aseon-api.onrender.com")

REPORT_TITLE = "Aseon — Monthly SEO/AEO/GEO Report"
REPORT_SECTIONS_DEFAULT = ["crawl", "keywords", "faq", "schema", "quick_wins", "plan"]
REPORT_FORMAT_DEFAULT = "markdown"  # 'markdown' | 'html' | 'both'


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_get(d: Dict[str, Any], path: List[Any], default=None):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def _bullet_list(items: List[str]) -> str:
    return "\n".join([f"- {i}" for i in items if i])


def _code_block(lang: str, content: str) -> str:
    return f"```{lang}\n{content}\n```"


def _section_heading(text: str, level: int = 2) -> str:
    return f"{'#' * level} {text}"


def _link(url: str, text: str) -> str:
    return f"[{text}]({url})"


def _latest_outputs_by_type(conn, site_id: str, types: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Haal per type het meest recente 'done' job-resultaat op.
    """
    if not types:
        return {}
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
        results: Dict[str, Dict[str, Any]] = {}
        for row in cur.fetchall():
            results[row["type"]] = {
                "id": row["id"],
                "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
                "output": row["output"] or {},
            }
        return results


def _render_markdown_report(title: str,
                            site_id: str,
                            latest: Dict[str, Dict[str, Any]],
                            include_sections: List[str]) -> str:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"_Generated at {_now_utc_iso()}_")
    lines.append("")

    # Crawl
    if "crawl" in include_sections and "crawl" in latest:
        crawl = latest["crawl"]
        cout = crawl["output"] or {}
        summary = _safe_get(cout, ["summary"], "")
        quick = _safe_get(cout, ["quick_wins"], []) or []
        issues = _safe_get(cout, ["issues"], []) or []
        jid = crawl["id"]
        lines.append(_section_heading("Crawl snapshot"))
        lines.append(f"Job: {_link(f'{API_BASE}/jobs/{jid}', jid)}")
        if summary:
            lines.append("")
            lines.append(str(summary).strip())
        if issues:
            lines.append("")
            lines.append("**Detected issues**")
            lines.append(_bullet_list([str(i) for i in issues]))
        if quick:
            lines.append("")
            lines.append("**Quick wins (from crawl)**")
            lines.append(_bullet_list([str(i) for i in quick]))
        lines.append("")

    # Keywords
    if "keywords" in include_sections and "keywords" in latest:
        kw = latest["keywords"]
        kout = kw["output"] or {}
        clusters: Dict[str, List[str]] = _safe_get(kout, ["clusters"], {}) or {}
        ideas: List[str] = _safe_get(kout, ["ideas"], []) or []
        jid = kw["id"]
        lines.append(_section_heading("Keyword ideas & intent clustering"))
        lines.append(f"Job: {_link(f'{API_BASE}/jobs/{jid}', jid)}")
        if clusters:
            lines.append("")
            lines.append("**Clusters**")
            for intent, terms in clusters.items():
                lines.append(f"- **{intent}**")
                lines.extend([f"  - {t}" for t in terms])
        if ideas:
            lines.append("")
            lines.append("**Additional ideas**")
            lines.extend([f"- {t}" for t in ideas])
        lines.append("")

    # FAQ
    if "faq" in include_sections and "faq" in latest:
        fq = latest["faq"]
        fout = fq["output"] or {}
        faqs = _safe_get(fout, ["faqs"], []) or []
        jid = fq["id"]
        lines.append(_section_heading("Quick answers (FAQ)"))
        lines.append(f"Job: {_link(f'{API_BASE}/jobs/{jid}', jid)}")
        for qa in faqs:
            q = str(qa.get("q", "")).strip()
            a = str(qa.get("a", "")).strip()
            src = qa.get("source") or qa.get("url") or ""
            if q:
                lines.append(f"- **Q:** {q}")
            if a:
                lines.append(f"  - **A:** {a}")
            if src:
                lines.append(f"  - Source: {src}")
        lines.append("")

    # Schema
    if "schema" in include_sections and "schema" in latest:
        sc = latest["schema"]
        sout = sc["output"] or {}
        jid = sc["id"]
        lines.append(_section_heading("Structured data (schema.org)"))
        lines.append(f"Job: {_link(f'{API_BASE}/jobs/{jid}', jid)}")
        schema_obj = sout.get("schema")
        if isinstance(schema_obj, dict) and schema_obj:
            pretty = json.dumps(schema_obj, indent=2, ensure_ascii=False)
            lines.append("")
            lines.append("Preview JSON-LD:")
            lines.append(_code_block("json", pretty))
        lines.append("")

    # Quick wins (geconsolideerd)
    if "quick_wins" in include_sections:
        items: List[str] = []
        if "crawl" in latest:
            items.extend(_safe_get(latest["crawl"]["output"] or {}, ["quick_wins"], []) or [])
        if "keywords" in latest:
            items.extend(_safe_get(latest["keywords"]["output"] or {}, ["quick_wins"], []) or [])
        if "schema" in latest:
            items.extend(_safe_get(latest["schema"]["output"] or {}, ["quick_wins"], []) or [])
        items = [str(i) for i in items if i]
        if items:
            lines.append(_section_heading("Quick wins (consolidated)"))
            lines.append(_bullet_list(items))
            lines.append("")

    # Monthly plan
    if "plan" in include_sections:
        plan: List[str] = []
        crawl_issues = []
        if "crawl" in latest:
            crawl_issues = _safe_get(latest["crawl"]["output"] or {}, ["issues"], []) or []
        if crawl_issues:
            plan.append("Fix top technical issues from crawl (titles/meta/robots/canonicals).")
        if "keywords" in latest:
            plan.append("Prioritize two clusters for content creation (top-intent + feasible).")
        if "faq" in latest:
            plan.append("Publish FAQ blocks on 2–3 key pages to target snippet/AI answers.")
        if "schema" in latest:
            plan.append("Validate JSON-LD (FAQ/Organization/Article) and monitor.")
        if not plan:
            plan.append("Review data freshness; rerun crawl/keywords/faq/schema for latest signals.")
        lines.append(_section_heading("Monthly plan"))
        lines.append(_bullet_list(plan))
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _render_html_report(title: str,
                        site_id: str,
                        latest: Dict[str, Dict[str, Any]],
                        include_sections: List[str]) -> str:
    md = _render_markdown_report(title, site_id, latest, include_sections)
    escaped = (
        md.replace("&", "&amp;")
          .replace("<", "&lt;")
          .replace(">", "&gt;")
    )
    html = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body{{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;line-height:1.5;padding:24px;max-width:900px;margin:0 auto;}}
pre{{background:#f6f8fa;padding:16px;border-radius:8px;overflow:auto}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
h1,h2,h3{{margin-top:1.2em}}
a{{text-decoration:none}}
</style>
<body>
<pre>{escaped}</pre>
</body>
</html>"""
    return html


def _build_report_output(site_id: str,
                         latest: Dict[str, Dict[str, Any]],
                         payload: Dict[str, Any]) -> Dict[str, Any]:
    fmt = (payload.get("format") or REPORT_FORMAT_DEFAULT).lower()
    title = payload.get("title") or REPORT_TITLE
    include_sections = payload.get("sections") or REPORT_SECTIONS_DEFAULT

    out: Dict[str, Any] = {
        "site_id": site_id,
        "title": title,
        "format": fmt,
        "sections": include_sections,
        "sources": {
            t: {
                "job_id": latest[t]["id"],
                "job_url": f"{API_BASE}/jobs/{latest[t]['id']}",
                "finished_at": latest[t]["finished_at"],
            } for t in latest
        },
    }

    if fmt in ("markdown", "both"):
        out["markdown"] = _render_markdown_report(title, site_id, latest, include_sections)
    if fmt in ("html", "both"):
        out["html"] = _render_html_report(title, site_id, latest, include_sections)

    return out


def generate_report(conn, job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hoofdfunctie aangeroepen door general_agent.
    Verwacht: conn (psycopg3-connection), job dict met 'site_id' en 'payload'.
    """
    site_id = job["site_id"]
    payload = job.get("payload") or {}

    want_types = payload.get("include_types") or ["crawl", "keywords", "faq", "schema"]
    latest = _latest_outputs_by_type(conn, site_id, want_types)

    # Skeleton blijft deterministisch aanwezig (zelfs als er nog geen eerdere jobs zijn)
    output = _build_report_output(site_id, latest, payload)
    return output
