#!/usr/bin/env python3
import os
import io
import json
import base64
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from psycopg.rows import dict_row
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER

API_BASE = os.getenv("API_BASE_URL", "https://aseon-api.onrender.com")

REPORT_TITLE_DEFAULT = "Monthly SEO/AEO/GEO Report"
REPORT_SECTIONS_DEFAULT = ["crawl", "keywords", "faq", "schema", "quick_wins", "plan"]
REPORT_FORMAT_DEFAULT = "markdown"  # "markdown" | "html" | "pdf" | "both"


# ---------- utils ----------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def _safe_get(d: Dict[str, Any], path: List[Any], default=None):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur

def _is_asset_url(u: str) -> bool:
    if not isinstance(u, str) or not u:
        return True
    if "#" in u:
        return True
    low = u.lower()
    if any(x in low for x in ("favicon", "apple-touch-icon")):
        return True
    if re.search(r"\.(png|jpg|jpeg|gif|svg|webp|ico|css|js|pdf)(\?|$)", low):
        return True
    return False


# ---------- data access (robust: 1 query per type; geen ANY/arrays) ----------

def _get_site(conn, site_id: str) -> Dict[str, Any]:
    site = {"url": "?", "language": "?", "country": "?"}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT url, language, country FROM sites WHERE id=%s",
            (site_id,),
        )
        row = cur.fetchone()
        if row:
            site = dict(row)
    return site

def _get_latest_job(conn, site_id: str, jtype: str) -> Dict[str, Any] | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, output, finished_at
              FROM jobs
             WHERE site_id=%s AND type=%s AND status='done'
             ORDER BY finished_at DESC NULLS LAST, created_at DESC
             LIMIT 1
            """,
            (site_id, jtype),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
            "output": row.get("output") or {},
        }

def _latest_outputs_by_type(conn, site_id: str, types: List[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for t in types:
        r = _get_latest_job(conn, site_id, t)
        if r:
            out[t] = r
    return out


# ---------- shaping helpers (sluiten aan op jouw agents) ----------

def _shape_crawl(cout: Dict[str, Any]) -> Dict[str, Any]:
    pages = cout.get("pages") or []
    summary = cout.get("summary") or {}
    pages_total = summary.get("pages_total") or len(pages) or 0
    ok_200 = summary.get("ok_200") or 0
    redir = summary.get("redirect_3xx") or 0
    err_45 = summary.get("errors_4xx_5xx") or 0
    dur_ms = summary.get("duration_ms")
    quick_wins = cout.get("quick_wins") or []  # list of {"type": ...}

    # Top issue pages (skip assets/anchors)
    issue_rows: List[str] = []
    for p in pages:
        url = p.get("final_url") or p.get("url") or ""
        if _is_asset_url(url):
            continue
        issues = [i for i in (p.get("issues") or []) if i]
        if not issues:
            continue
        issue_rows.append(f"{url} — {', '.join(issues)}")
    issue_rows = issue_rows[:10]

    # Quick win counts
    qcount: Dict[str, int] = {}
    for q in quick_wins:
        t = q.get("type")
        if not t:
            continue
        qcount[t] = qcount.get(t, 0) + 1

    return {
        "pages_total": pages_total,
        "ok_200": ok_200,
        "redirect_3xx": redir,
        "errors_4xx_5xx": err_45,
        "duration_ms": dur_ms,
        "issue_rows": issue_rows,
        "quick_win_counts": qcount,
    }

def _shape_keywords(kout: Dict[str, Any]) -> Tuple[Dict[str, List[str]], List[str]]:
    clusters = kout.get("clusters") or {}
    # Normaliseer: verwacht dict intent->list
    shaped: Dict[str, List[str]] = {}
    for intent, terms in clusters.items():
        if isinstance(terms, list):
            shaped[str(intent)] = [str(t) for t in terms if isinstance(t, (str, int, float))][:12]
        elif isinstance(terms, dict):
            shaped[str(intent)] = [str(k) for k in list(terms.keys())][:12]
        else:
            shaped[str(intent)] = [str(terms)]
    suggestions = kout.get("suggestions") or []
    sug_lines: List[str] = []
    for s in suggestions[:3]:
        if isinstance(s, str):
            sug_lines.append(s)
        elif isinstance(s, dict):
            title = s.get("title") or s.get("name") or "Suggested page"
            terms = s.get("terms") or s.get("keywords") or []
            if isinstance(terms, dict):
                terms = list(terms.keys())
            if isinstance(terms, list):
                terms = ", ".join([str(t) for t in terms[:6]])
            sug_lines.append(f"{title} — {terms}")
    return shaped, sug_lines

def _shape_faq(fout: Dict[str, Any]) -> List[Dict[str, str]]:
    faqs = fout.get("faqs") or []
    shaped = []
    for qa in faqs[:5]:
        q = str(qa.get("q") or "").strip()
        a = str(qa.get("a") or "").strip()
        src = qa.get("source") or None
        if not q or not a:
            continue
        shaped.append({"q": q, "a": a, "source": src})
    return shaped


# ---------- renderers ----------

def _render_markdown(title: str, site: Dict[str, Any], latest: Dict[str, Any], include: List[str]) -> str:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Site:** {site.get('url')}  \n**Language/Country:** {site.get('language')}/{site.get('country')}  \n_Generated: {_now_iso()}_")
    lines.append("")

    # Crawl
    if "crawl" in include and "crawl" in latest:
        c = _shape_crawl(latest["crawl"]["output"] or {})
        lines.append("## Crawl summary")
        lines.append(f"- Pages: **{c['pages_total']}**  ·  200={c['ok_200']}  ·  3xx={c['redirect_3xx']}  ·  4xx/5xx={c['errors_4xx_5xx']}")
        if c["duration_ms"] is not None:
            lines.append(f"- Duration: ~{int(c['duration_ms'])/1000:.1f}s")
        if c["issue_rows"]:
            lines.append("**Pages with issues (top 10):**")
            for r in c["issue_rows"]:
                lines.append(f"- {r}")
        if c["quick_win_counts"]:
            lines.append("")
            lines.append("**Quick wins (counts):**")
            for k, v in sorted(c["quick_win_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"- **{k}**: {v}")
        lines.append("")

    # Keywords
    if "keywords" in include and "keywords" in latest:
        kout = latest["keywords"]["output"] or {}
        clusters, sugg = _shape_keywords(kout)
        if clusters:
            lines.append("## Keyword clusters")
            for intent, terms in clusters.items():
                show = ", ".join(terms[:8])
                lines.append(f"- **{intent}**: {show}{'…' if len(terms) > 8 else ''}")
            lines.append("")
        if sugg:
            lines.append("**Suggested pages:**")
            for s in sugg:
                lines.append(f"- {s}")
            lines.append("")

    # FAQ
    if "faq" in include and "faq" in latest:
        faqs = _shape_faq(latest["faq"]["output"] or {})
        if faqs:
            lines.append("## FAQs (≤80 words)")
            for qa in faqs:
                lines.append(f"- **Q:** {qa['q']}")
                lines.append(f"  - A: {qa['a']}" + (f"  [(source)]({qa['source']})" if qa.get("source") else ""))
            lines.append("")

    # Schema
    if "schema" in include and "schema" in latest:
        s = latest["schema"]["output"] or {}
        schema = s.get("schema")
        lines.append("## Schema JSON-LD (preview)")
        if isinstance(schema, dict):
            try:
                lines.append("```json")
                lines.append(json.dumps(schema, indent=2, ensure_ascii=False))
                lines.append("```")
            except Exception:
                lines.append("```json")
                lines.append(json.dumps(schema))
                lines.append("```")
        lines.append("")

    # Consolidated quick wins
    if "quick_wins" in include:
        agg: Dict[str, int] = {}
        if "crawl" in latest:
            c = _shape_crawl(latest["crawl"]["output"] or {})
            for k, v in (c["quick_win_counts"] or {}).items():
                agg[k] = agg.get(k, 0) + v
        if agg:
            lines.append("## Quick wins (consolidated)")
            for k, v in sorted(agg.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"- **{k}**: {v}")
            lines.append("")

    # Plan
    if "plan" in include:
        plan: List[str] = []
        if "crawl" in latest:
            plan.append("Fix top technical issues (titles, meta, robots, canonicals).")
        if "keywords" in latest:
            plan.append("Publish 2 pages for strongest keyword clusters.")
        if "faq" in latest:
            plan.append("Add FAQ blocks on key pages to target snippets/AI answers.")
        if "schema" in latest:
            plan.append("Validate JSON-LD in Search Console / Rich Results Test.")
        if not plan:
            plan.append("Prime the pipeline: run crawl, keywords, faq, schema this week.")
        lines.append("## Monthly plan")
        for p in plan:
            lines.append(f"- {p}")

    # Job links
    if latest:
        lines.append("")
        lines.append("## Job links")
        for t in ("crawl", "keywords", "faq", "schema"):
            if t in latest:
                jid = latest[t]["id"]
                ts = latest[t]["finished_at"]
                lines.append(f"- **{t}** — [{jid}]({API_BASE}/jobs/{jid}) · {ts}")

    return "\n".join(lines)


def _render_pdf(title: str, site: Dict[str, Any], latest: Dict[str, Any], include: List[str]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=title)
    styles = getSampleStyleSheet()
    story = []

    tstyle = styles["Title"]; tstyle.alignment = TA_CENTER
    story.append(Paragraph(title, tstyle))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Site: {site.get('url')} ({site.get('language')}/{site.get('country')})", styles["Normal"]))
    story.append(Paragraph(f"Generated: {_now_iso()}", styles["Normal"]))
    story.append(Spacer(1, 18))

    def add_section(head: str, items: List[str]):
        story.append(Paragraph(head, styles["Heading2"]))
        story.append(Spacer(1, 6))
        if items:
            lf = ListFlowable([ListItem(Paragraph(i, styles["Normal"])) for i in items], bulletType="bullet")
            story.append(lf)
        story.append(Spacer(1, 12))

    # Crawl
    if "crawl" in include and "crawl" in latest:
        c = _shape_crawl(latest["crawl"]["output"] or {})
        items = [f"Pages: {c['pages_total']} · 200={c['ok_200']} · 3xx={c['redirect_3xx']} · 4xx/5xx={c['errors_4xx_5xx']}"]
        if c["duration_ms"] is not None:
            items.append(f"Duration: ~{int(c['duration_ms'])/1000:.1f}s")
        items += c["issue_rows"]
        if c["quick_win_counts"]:
            for k, v in sorted(c["quick_win_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
                items.append(f"Quick win: {k} — {v}")
        add_section("Crawl summary", items)

    # Keywords
    if "keywords" in include and "keywords" in latest:
        kout = latest["keywords"]["output"] or {}
        clusters, sugg = _shape_keywords(kout)
        items = [f"{intent}: {', '.join(terms[:8])}{'…' if len(terms) > 8 else ''}" for intent, terms in clusters.items()]
        if sugg:
            items += [f"Suggestion: {s}" for s in sugg]
        add_section("Keyword clusters", items)

    # FAQs
    if "faq" in include and "faq" in latest:
        faqs = _shape_faq(latest["faq"]["output"] or {})
        items = [f"Q: {qa['q']} — A: {qa['a']}" + (f" (source: {qa['source']})" if qa.get("source") else "") for qa in faqs]
        add_section("FAQs", items)

    # Schema
    if "schema" in include and "schema" in latest:
        add_section("Schema (JSON-LD)", ["See API output for full JSON."])

    # Quick wins consolidated
    if "quick_wins" in include:
        agg: Dict[str, int] = {}
        if "crawl" in latest:
            c = _shape_crawl(latest["crawl"]["output"] or {})
            for k, v in (c["quick_win_counts"] or {}).items():
                agg[k] = agg.get(k, 0) + v
        items = [f"{k}: {v}" for k, v in sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))]
        add_section("Quick wins", items)

    # Plan
    if "plan" in include:
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
    include = payload.get("sections") or REPORT_SECTIONS_DEFAULT
    title = payload.get("title") or REPORT_TITLE_DEFAULT

    latest = _latest_outputs_by_type(conn, site_id, ["crawl", "keywords", "faq", "schema"])
    site = _get_site(conn, site_id)

    out: Dict[str, Any] = {
        "site": site,
        "title": title,
        "generated_at": _now_iso(),
        "format": fmt,
    }

    # Always markdown
    md = _render_markdown(title, site, latest, include)
    out["markdown"] = md

    # Optional HTML
    if fmt in ("html", "both"):
        out["html"] = f"<pre>{md}</pre>"

    # Optional PDF
    if fmt in ("pdf", "both"):
        pdf_bytes = _render_pdf(title, site, latest, include)
        out["pdf_base64"] = base64.b64encode(pdf_bytes).decode("utf-8")

    # Job references (handig voor UI)
    jobs_meta = {}
    for t in ("crawl", "keywords", "faq", "schema"):
        if t in latest:
            jobs_meta[t] = {
                "id": str(latest[t]["id"]),
                "url": f"{API_BASE}/jobs/{latest[t]['id']}",
                "finished_at": latest[t]["finished_at"],
            }
    if jobs_meta:
        out["jobs"] = jobs_meta

    return out
