# report_agent.py
import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from psycopg.rows import dict_row
from psycopg import Connection

# ===== Config & Defaults =====
API_BASE_URL = os.getenv("ASEON_API_BASE_URL", "https://aseon-api.onrender.com").rstrip("/")
DEFAULT_REPORT_FORMAT = os.getenv("ASEON_REPORT_FORMAT", "markdown")  # "markdown" | "html" | "both"
MAX_LIST_ITEMS = int(os.getenv("REPORT_MAX_LIST_ITEMS", "10"))        # per sectie limiter
DATE_FMT = os.getenv("REPORT_DATE_FMT", "%Y-%m-%d %H:%M:%S %Z")


# ===== DB helpers (self-contained; geen afhankelijkheid van general_agent) =====
def _get_site_info(conn: Connection, site_id: str) -> Dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT s.id, s.url, s.language, s.country, a.name AS account_name, a.id AS account_id
              FROM sites s
              JOIN accounts a ON a.id = s.account_id
             WHERE s.id = %s
        """, (site_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError("Site not found")
        return row


def _get_latest_job(conn: Connection, site_id: str, jtype: str) -> Optional[Dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT id, type, output, finished_at, created_at, status, error
              FROM jobs
             WHERE site_id=%s AND type=%s AND status='done'
             ORDER BY COALESCE(finished_at, created_at) DESC
             LIMIT 1
        """, (site_id, jtype))
        row = cur.fetchone()
        return row if row else None


def _fmt_dt(dt: Optional[datetime]) -> str:
    if not isinstance(dt, datetime):
        return "-"
    # Normaliseer naar UTC voor consistentie; UI kan lokaal formatteren
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime(DATE_FMT.replace("%Z", "UTC"))


def _link_for_job(job_id: str) -> str:
    # Direct deep-link naar API detail (handig in dashboard / externe tooling)
    return f"{API_BASE_URL}/jobs/{job_id}"


# ===== Render helpers =====
def _safe_json(obj: Any, pretty: bool = True) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2 if pretty else None)
    except Exception:
        return json.dumps({"_aseon": "unserializable"})


def _mk_crawl_section_md(latest: Optional[Dict[str, Any]]) -> Tuple[str, list]:
    if not latest or not latest.get("output"):
        return "Geen crawl-gegevens beschikbaar.\n", []
    out = latest["output"] or {}
    summary = out.get("summary") or {}
    quick_wins = out.get("quick_wins") or []

    lines = []
    lines.append(f"- Pagina’s gecrawld: **{summary.get('pages_total', 0)}**")
    lines.append(
        f"- Status: 200={summary.get('ok_200', 0)}, 3xx={summary.get('redirect_3xx', 0)}, 4xx/5xx={summary.get('errors_4xx_5xx', 0)}"
    )
    dur_ms = summary.get("duration_ms")
    if isinstance(dur_ms, int):
        lines.append(f"- Duur: ~{int(dur_ms/1000)}s")

    # Top pages met issues (max MAX_LIST_ITEMS)
    pages = out.get("pages") or []
    with_issues = [p for p in pages if p.get("issues")]
    top = with_issues[:MAX_LIST_ITEMS]
    if top:
        lines.append("\n**Pagina’s met snelle kansen (issues):**")
        for p in top:
            url = p.get("final_url") or p.get("url") or "-"
            issues = ", ".join(p.get("issues") or [])
            lines.append(f"  - {url} — _{issues}_")

    # Quick wins (geaggregeerd)
    if quick_wins:
        lines.append("\n**Quick wins (geaggregeerd):**")
        for qw in quick_wins[:MAX_LIST_ITEMS]:
            t = qw.get("type") or "issue"
            lines.append(f"  - {t}")

    return "\n".join(lines) + "\n", quick_wins


def _mk_keywords_section_md(latest: Optional[Dict[str, Any]]) -> str:
    if not latest or not latest.get("output"):
        return "Geen keyword-gegevens beschikbaar.\n"
    out = latest["output"] or {}
    clusters = (out.get("clusters") or {})
    suggestions = out.get("suggestions") or []
    seed = out.get("seed")

    lines = []
    if seed:
        lines.append(f"_Seed:_ `{seed}`")
    if clusters:
        lines.append("\n**Clusters (intent):**")
        for key in ("informational", "transactional", "navigational"):
            arr = clusters.get(key) or []
            if arr:
                shown = ", ".join(arr[:8])
                more = f" …(+{max(0, len(arr)-8)})" if len(arr) > 8 else ""
                lines.append(f"- **{key.capitalize()}**: {shown}{more}")
    if suggestions:
        lines.append("\n**Suggesties voor pagina’s:**")
        for s in suggestions[:MAX_LIST_ITEMS]:
            if isinstance(s, str):
                lines.append(f"  - {s}")
            elif isinstance(s, dict):
                title = s.get("title") or s.get("name") or "Suggestie"
                kws = s.get("keywords") or s.get("terms") or []
                kw_txt = ", ".join(kws[:6]) + (" …" if len(kws) > 6 else "")
                lines.append(f"  - **{title}** — {kw_txt}")
    return "\n".join(lines) + "\n"


def _mk_faq_section_md(latest: Optional[Dict[str, Any]]) -> str:
    if not latest or not latest.get("output"):
        return "Geen FAQ-gegevens beschikbaar.\n"
    out = latest["output"] or {}
    faqs = out.get("faqs") or []
    if not faqs:
        return "Geen FAQ-gegevens beschikbaar.\n"

    lines = []
    for f in faqs[:MAX_LIST_ITEMS]:
        q = f.get("q") or "Vraag"
        a = (f.get("a") or "").strip()
        src = f.get("source")
        src_txt = f" [bron]({src})" if src else ""
        lines.append(f"- **{q}** — {a}{src_txt}")
    return "\n".join(lines) + "\n"


def _mk_schema_section_md(latest: Optional[Dict[str, Any]]) -> str:
    if not latest or not latest.get("output"):
        return "Geen schema-gegevens beschikbaar.\n"
    out = latest["output"] or {}
    schema_obj = out.get("schema") or out  # fallback, voor het geval output direct het object is
    # Compact maar leesbaar
    schema_str = _safe_json(schema_obj, pretty=True)
    return "```json\n" + schema_str + "\n```\n"


def _mk_job_links_md(rows: Dict[str, Optional[Dict[str, Any]]]) -> str:
    lines = []
    for t in ("crawl", "keywords", "faq", "schema"):
        row = rows.get(t)
        if row and row.get("id"):
            link = _link_for_job(str(row["id"]))
            ts = _fmt_dt(row.get("finished_at"))
            lines.append(f"- **{t}** — [{row['id']}]({link}) · {ts}")
        else:
            lines.append(f"- **{t}** — _n/a_")
    return "\n".join(lines) + "\n"


def _build_markdown(site: Dict[str, Any], rows: Dict[str, Optional[Dict[str, Any]]]) -> Tuple[str, Dict[str, Any]]:
    crawl_md, quick_wins = _mk_crawl_section_md(rows.get("crawl"))
    kws_md = _mk_keywords_section_md(rows.get("keywords"))
    faq_md = _mk_faq_section_md(rows.get("faq"))
    schema_md = _mk_schema_section_md(rows.get("schema"))
    links_md = _mk_job_links_md(rows)

    generated_at = datetime.now(timezone.utc).strftime(DATE_FMT.replace("%Z", "UTC"))

    header = [
        f"# Aseon Report — {site.get('account_name') or ''}".strip(),
        f"- **Site:** {site.get('url')}",
        f"- **Taal/Land:** {site.get('language')}/{site.get('country')}",
        f"- **Gegenereerd:** {generated_at}",
        "",
    ]

    body = [
        "## Crawl summary",
        crawl_md,
        "## Keyword clusters",
        kws_md,
        "## FAQ’s (≤80 woorden, met bron)",
        faq_md,
        "## Schema JSON-LD snippet",
        schema_md,
        "## Quick wins (samenvatting)",
    ]

    # Quick wins extra samenvatting
    if quick_wins:
        types = [qw.get("type") or "issue" for qw in quick_wins]
        # tel per type
        counts: Dict[str, int] = {}
        for t in types:
            counts[t] = counts.get(t, 0) + 1
        wins_lines = [f"- **{k}**: {v}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        body.append("\n".join(wins_lines) + "\n")
    else:
        body.append("_Geen quick wins gevonden._\n")

    footer = [
        "## Job links (details)",
        links_md
    ]

    md = "\n".join(header + body + footer).strip() + "\n"
    meta = {
        "site": {
            "id": str(site.get("id")),
            "url": site.get("url"),
            "language": site.get("language"),
            "country": site.get("country"),
            "account_id": str(site.get("account_id")),
            "account_name": site.get("account_name"),
        },
        "generated_at": generated_at,
    }
    return md, meta


def _escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )


def _build_html(site: Dict[str, Any], rows: Dict[str, Optional[Dict[str, Any]]], md_text: Optional[str] = None) -> str:
    """
    Simpele, dependency-vrije HTML (geen markdown parser nodig).
    We renderen secties opnieuw in HTML voor nette opmaak.
    """
    # Rebuild secties voor nette HTML:
    crawl_md, quick_wins = _mk_crawl_section_md(rows.get("crawl"))
    kws_md = _mk_keywords_section_md(rows.get("keywords"))
    faq_md = _mk_faq_section_md(rows.get("faq"))
    schema_md = _mk_schema_section_md(rows.get("schema"))
    links_md = _mk_job_links_md(rows)
    generated_at = datetime.now(timezone.utc).strftime(DATE_FMT.replace("%Z", "UTC"))

    def pblock(md: str) -> str:
        # elke regel -> <p> of <li> binnen <ul> wanneer '- ' of '  - ' prefix
        html_lines = []
        for line in md.strip().splitlines():
            if line.strip().startswith("- "):
                html_lines.append(f"<li>{_escape_html(line.strip()[2:])}</li>")
            elif line.strip().startswith("  - "):
                html_lines.append(f"<li style='margin-left:1rem'>{_escape_html(line.strip()[4:])}</li>")
            else:
                if line.strip():
                    html_lines.append(f"<p>{_escape_html(line)}</p>")
        # wrap in <ul> als er <li>’s zijn
        if any(l.startswith("<li") for l in html_lines):
            # scheid paragrafen en list
            items = [l for l in html_lines if l.startswith("<li")]
            paras = [l for l in html_lines if l.startswith("<p")]
            out = ""
            if paras:
                out += "".join(paras)
            if items:
                out += "<ul>" + "".join(items) + "</ul>"
            return out
        return "".join(html_lines)

    schema_code = _escape_html(schema_md.replace("```json", "").replace("```", "").strip())

    return f"""<!doctype html>
<html lang="nl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Aseon Report — { _escape_html(site.get('account_name') or '') }</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji"; line-height:1.55; padding: 24px; color:#0f172a; }}
    h1 {{ font-size: 1.8rem; margin:0 0 0.25rem; }}
    h2 {{ font-size: 1.25rem; margin:1.5rem 0 0.5rem; }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
    pre {{ background:#0b1020; color:#e2e8f0; padding:12px; border-radius:12px; overflow:auto; }}
    .meta small {{ color:#475569; }}
    ul {{ margin:0.25rem 0 0.75rem 1.25rem; }}
  </style>
</head>
<body>
  <h1>Aseon Report — { _escape_html(site.get('account_name') or '') }</h1>
  <div class="meta">
    <p><strong>Site:</strong> { _escape_html(site.get('url') or '-') }<br>
    <strong>Taal/Land:</strong> { _escape_html((site.get('language') or '-') + '/' + (site.get('country') or '-')) }<br>
    <small>Gegenereerd: { _escape_html(generated_at) }</small></p>
  </div>

  <h2>Crawl summary</h2>
  { pblock(crawl_md) }

  <h2>Keyword clusters</h2>
  { pblock(kws_md) }

  <h2>FAQ’s (≤80 woorden, met bron)</h2>
  { pblock(faq_md) }

  <h2>Schema JSON-LD snippet</h2>
  <pre><code>{ schema_code }</code></pre>

  <h2>Quick wins (samenvatting)</h2>
  { pblock(_mk_crawl_section_md(rows.get("crawl"))[0].split("**Quick wins (geaggregeerd):**")[-1] if "Quick wins" in crawl_md else "_Geen quick wins gevonden._") }

  <h2>Job links (details)</h2>
  { pblock(links_md) }
</body>
</html>
"""


# ===== Public API =====
def generate_report(conn: Connection, site_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Bouwt een rapport op basis van de laatste resultaten.
    Return-structuur is geschikt om direct door general_agent.finish_job() in jobs.output te worden weggeschreven.
    """
    payload = payload or {}
    fmt = (payload.get("format") or DEFAULT_REPORT_FORMAT or "markdown").lower().strip()
    if fmt not in ("markdown", "html", "both"):
        fmt = "markdown"

    site = _get_site_info(conn, site_id)

    # Haal laatste jobs op en bewaar id + output + finished_at (voor links)
    rows: Dict[str, Optional[Dict[str, Any]]] = {}
    for t in ("crawl", "keywords", "faq", "schema"):
        row = _get_latest_job(conn, site_id, t)
        rows[t] = row

    # Bouw Markdown (en optioneel HTML)
    md, meta = _build_markdown(site, rows)
    html = _build_html(site, rows, md_text=md) if fmt in ("html", "both") else None

    # Bouw job-links map voor API/dashboards
    job_links: Dict[str, Optional[Dict[str, str]]] = {}
    for t, row in rows.items():
        if row and row.get("id"):
            job_links[t] = {
                "id": str(row["id"]),
                "url": _link_for_job(str(row["id"])),
                "finished_at": _fmt_dt(row.get("finished_at")),
            }
        else:
            job_links[t] = None

    out: Dict[str, Any] = {
        "format": fmt if fmt != "both" else "markdown+html",
        "report": md if fmt in ("markdown", "both") else None,
        "html": html if fmt in ("html", "both") else None,
        "jobs": job_links,
        "site": meta["site"],
        "generated_at": meta["generated_at"],
        "_aseon": {
            "version": "1.0.0",
            "source": "report_agent.py",
            "api_base": API_BASE_URL
        }
    }
    # Always non-empty guarantee (defensief; general_agent.finish_job verwacht dict)
    if not out.get("report") and not out.get("html"):
        out["report"] = "# Aseon Report\n\n_Geen data beschikbaar._\n"
    return out


# ===== CLI-hook (optioneel voor lokale testen) =====
if __name__ == "__main__":
    # Voor snelle handmatige checks kun je ASEON_DSN zetten en een SITE_ID doorgeven.
    # Voorbeeld:
    #   ASEON_DSN=postgresql://... python report_agent.py
    import os
    import psycopg

    dsn = os.getenv("ASEON_DSN") or os.getenv("DATABASE_URL")
    site_id = os.getenv("SITE_ID")
    if not dsn or not site_id:
        print("Set ASEON_DSN/DATABASE_URL en SITE_ID voor een lokale test.", flush=True)
        raise SystemExit(1)

    with psycopg.connect(dsn) as conn:
        result = generate_report(conn, site_id, {"format": "both"})
        # Schrijf niet terug naar DB in CLI-mode; print naar stdout
        print(json.dumps(result, ensure_ascii=False)[:2000], flush=True)
