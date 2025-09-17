# report_agent.py — Full report (SEO + GEO + AEO) with LLM narrative + deterministic fallback
# Returns: {"html_base64": ..., "patches_csv_base64": ..., "plan_csv_base64": ...}

import os
import base64
import html
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from psycopg.rows import dict_row

# ========= Utilities =========

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def _escape_ascii(s: str) -> str:
    if s is None:
        return ""
    t = html.unescape(str(s))
    repl = {
        "\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00A0": " ",
        "\u200b": ""
    }
    for k, v in repl.items():
        t = t.replace(k, v)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _h(s: str) -> str:
    return html.escape(_escape_ascii(s or ""))

def _csv_escape(s: str) -> str:
    s = _escape_ascii(s)
    if '"' in s or "," in s or "\n" in s:
        return '"' + s.replace('"', '""') + '"'
    return s

def _fetch_latest(conn, site_id: str, jtype: str) -> Optional[Dict[str, Any]]:
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

def _norm_url(u: str) -> str:
    try:
        p = urlparse((u or "").strip())
        scheme = p.scheme or "https"
        host = p.netloc.lower()
        path = p.path or "/"
        query = p.query
        return f"{scheme}://{host}{path}" + (f"?{query}" if query else "")
    except Exception:
        return (u or "").strip()

def _host(u: str) -> str:
    try:
        return urlparse(u).netloc.lower()
    except Exception:
        return ""

def _first_nonempty(arr: List[str]) -> str:
    for x in arr or []:
        if (x or "").strip():
            return x.strip()
    return ""

def _trim_words(s: str, n: int) -> str:
    words = _escape_ascii(s).split()
    if len(words) <= n:
        return " ".join(words)
    return " ".join(words[:n]) + "…"

# ========= Crawl → SEO analysis =========

def _len_ok_title(s: str) -> bool:
    l = len(_escape_ascii(s))
    return 45 <= l <= 65

def _len_ok_meta(s: str) -> bool:
    l = len(_escape_ascii(s))
    return 120 <= l <= 170

def _best_meta_desc_from_page(page: Dict[str, Any]) -> str:
    par = page.get("paragraphs") or []
    txt = " ".join([p for p in par if p][:3]).strip()
    if not txt:
        h2 = page.get("h2") or []
        h3 = page.get("h3") or []
        li = page.get("li") or []
        txt = " ".join((h2[:1] + h3[:1] + li[:2])).strip()
    txt = _escape_ascii(txt) or (page.get("title") or "")
    return _trim_words(txt, 28)

def _propose_title(page: Dict[str, Any], brand: str) -> str:
    title = (page.get("title") or "").strip()
    h1 = (page.get("h1") or "").strip()
    cand = _escape_ascii(h1 or title) or _escape_ascii(_first_nonempty(page.get("h2") or [])) or "Page"
    if len(cand) < 38 and brand and brand.lower() not in cand.lower():
        cand = f"{cand} | {brand}"
    if len(cand) > 66:
        cand = _trim_words(cand, 10)
    return cand

def _analyze_seo_from_crawl(crawl: Dict[str, Any]) -> Dict[str, Any]:
    pages = crawl.get("pages") or []
    fixes = []
    canonical_patches = []
    og_patches = []
    titles_map = {}
    descs_map = {}

    for p in pages:
        url = _norm_url(p.get("url") or "")
        if not url:
            continue
        meta = p.get("meta") or {}
        title = p.get("title") or ""
        desc = meta.get("description") or ""
        canonical = p.get("canonical") or ""
        og_t = meta.get("og:title") or ""
        og_d = meta.get("og:description") or ""

        titles_map.setdefault(_escape_ascii(title), []).append(url)
        if desc:
            descs_map.setdefault(_escape_ascii(desc), []).append(url)

        if not _len_ok_title(title):
            proposed = _propose_title(p, brand=_host(url).split(":")[0])
            fixes.append({
                "url": url, "field": "title", "issue": "length suboptimal" if title else "missing",
                "current": title or "(empty)", "proposed": proposed
            })

        if not _len_ok_meta(desc):
            issue = "missing" if not desc else "length suboptimal"
            proposed_desc = _best_meta_desc_from_page(p)
            fixes.append({
                "url": url, "field": "meta_description", "issue": issue,
                "current": desc or "(empty)", "proposed": proposed_desc
            })

        if canonical:
            c_norm = _norm_url(canonical)
            if c_norm != url:
                canonical_patches.append({
                    "url": url, "category": "canonical", "issue": "differs from page URL",
                    "current": canonical, "patch": f'<link rel="canonical" href="{url}">'
                })
        else:
            canonical_patches.append({
                "url": url, "category": "canonical", "issue": "missing",
                "current": "(none)", "patch": f'<link rel="canonical" href="{url}">'
            })

        if not og_t or not og_d:
            t_val = _escape_ascii(title) or _propose_title(p, brand=_host(url))
            d_val = _escape_ascii(desc) or _best_meta_desc_from_page(p)
            og_patch = []
            if not og_t:
                og_patch.append(f'<meta property="og:title" content="{html.escape(t_val)}">')
            if not og_d:
                og_patch.append(f'<meta property="og:description" content="{html.escape(d_val)}">')
            og_patches.append({
                "url": url, "category": "open_graph",
                "issue": "missing og:title/og:description" if (not og_t and not og_d)
                         else ("missing og:title" if not og_t else "missing og:description"),
                "current": f"og:title={'missing' if not og_t else 'ok'}, og:description={'missing' if not og_d else 'ok'}",
                "patch": "\n".join(og_patch)
            })

    dup_titles = {k: v for k, v in titles_map.items() if k and len(v) > 1}
    dup_descs  = {k: v for k, v in descs_map.items() if k and len(v) > 1}

    return {
        "fixes": fixes,
        "canonical_patches": canonical_patches,
        "og_patches": og_patches,
        "dup_titles": dup_titles,
        "dup_descs": dup_descs
    }

# ========= AEO helpers =========

def _emit_aeo_scorecards(pages_aeo: List[Dict[str, Any]]) -> str:
    rows = []
    rows.append("<tr><th>Page URL</th><th>Type</th><th>Score (0–100)</th><th>Issues</th><th>Metrics</th></tr>")
    for p in pages_aeo:
        url = _h(p.get("url",""))
        ptype = _h(p.get("type","other"))
        score = str(p.get("score", 0))
        issues = ", ".join(p.get("issues") or []) or "OK"
        m = p.get("metrics") or {}
        src_counts = m.get("src_counts") or {}
        metrics_txt = f"qa_ok:{m.get('answers_leq_80w',0)}, parity_ok:{bool(m.get('parity_ok'))}, src_counts:{src_counts}, qas_detected:{m.get('qas_detected',0)}, has_faq_schema_detected:{'Yes' if m.get('has_faq_schema_detected') else ''}"
        rows.append(f"<tr><td><a href=\"{url}\">{url}</a></td><td>[{ptype}]</td><td>{score}</td><td>{_h(issues)}</td><td><span class='mono'>{_h(metrics_txt)}</span></td></tr>")
    return "<table class='grid'>" + "\n".join(rows) + "</table>"

def _emit_aeo_qna(pages_aeo: List[Dict[str, Any]]) -> str:
    rows = []
    rows.append("<tr><th>Page URL</th><th>Q</th><th>Proposed answer (≤80 words)</th></tr>")
    for p in pages_aeo:
        qas = p.get("qas") or []
        if not qas:
            continue
        url = _h(p.get("url",""))
        for qa in qas:
            q = _h(qa.get("q",""))
            a = _h(qa.get("a",""))
            rows.append(f"<tr><td><a href=\"{url}\">{url}</a></td><td>{q}</td><td>{a}</td></tr>")
    return "<table class='grid'>" + "\n".join(rows) + "</table>"

# ========= SEO sections from crawl analysis =========

def _emit_seo_fixes(fixes: List[Dict[str, str]]) -> str:
    if not fixes:
        return "<p>No concrete SEO text fixes found.</p>"
    rows = []
    rows.append("<tr><th>Page URL</th><th>Field</th><th>Issue</th><th>Current</th><th>Proposed (ready-to-paste)</th></tr>")
    for f in fixes:
        rows.append(
            f"<tr><td><a href=\"{_h(f['url'])}\">{_h(f['url'])}</a></td>"
            f"<td>{_h(f['field'])}</td><td>{_h(f['issue'])}</td>"
            f"<td>{_h(f['current'])}</td><td>{_h(f['proposed'])}</td></tr>"
        )
    return "<table class='grid'>" + "\n".join(rows) + "</table>"

def _emit_html_patches(canon: List[Dict[str, str]], ogs: List[Dict[str,str]]) -> str:
    lines = []
    if canon:
        lines.append("<h3>HTML patches — Canonical & Open Graph</h3>")
        rows = []
        rows.append("<tr><th>Page URL</th><th>Category</th><th>Issue</th><th>Current</th><th>Patch (copy/paste)</th></tr>")
        for c in canon:
            rows.append(
                f"<tr><td><a href=\"{_h(c['url'])}\">{_h(c['url'])}</a></td>"
                f"<td>{_h(c['category'])}</td><td>{_h(c['issue'])}</td>"
                f"<td>{_h(c['current'])}</td><td><pre>{_h(c['patch'])}</pre></td></tr>"
            )
        lines.append("<table class='grid'>" + "\n".join(rows) + "</table>")
    if ogs:
        rows = []
        rows.append("<tr><th>Page URL</th><th>Category</th><th>Issue</th><th>Current</th><th>Patch (copy/paste)</th></tr>")
        for o in ogs:
            rows.append(
                f"<tr><td><a href=\"{_h(o['url'])}\">{_h(o['url'])}</a></td>"
                f"<td>{_h(o['category'])}</td><td>{_h(o['issue'])}</td>"
                f"<td>{_h(o['current'])}</td><td><pre>{_h(o['patch'])}</pre></td></tr>"
            )
        lines.append("<table class='grid'>" + "\n".join(rows) + "</table>")
    return "\n".join(lines) if lines else "<p>No HTML patches needed.</p>"

def _emit_duplicate_groups(dups: Dict[str, List[str]], label: str) -> str:
    if not dups:
        return f"<p>No duplicate {label} found.</p>"
    rows = []
    rows.append("<tr><th>Value</th><th>URLs</th></tr>")
    for val, urls in dups.items():
        urls_html = "<br>".join(f"<a href=\"{_h(u)}\">{_h(u)}</a>" for u in urls)
        rows.append(f"<tr><td>{_h(val)}</td><td>{urls_html}</td></tr>")
    return "<table class='grid'>" + "\n".join(rows) + "</table>"

# ========= Implementation plan =========

def _add_task(tasks: List[Dict[str, Any]], url: str, task: str, why: str, effort: str, priority: float, has_patch: bool):
    tasks.append({
        "status": "☐",
        "url": url,
        "task": task,
        "why": why,
        "effort": effort,
        "priority": priority,
        "patch": "yes" if has_patch else "—"
    })

def _build_plan(crawl_analysis: Dict[str, Any], fixes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for c in crawl_analysis.get("canonical_patches") or []:
        _add_task(tasks, c["url"], "Fix canonical", "Avoids duplicate/canonical mismatch; improves indexing.", "S", 7.0, True)
    for f in fixes:
        if f["field"] == "title":
            _add_task(tasks, f["url"], "Optimize title (unique + in-range)", "Improves CTR and clarity; avoids duplicates.", "S", 6.0, True)
    for f in fixes:
        if f["field"] == "meta_description":
            _add_task(tasks, f["url"], "Add/optimize meta description", "Improves snippet quality and click-through.", "S", 6.0, True)
    for o in crawl_analysis.get("og_patches") or []:
        _add_task(tasks, o["url"], "Set up Open Graph", "Cleaner social/AI previews; indirect CTR benefits.", "S", 3.0, True)
    return tasks

def _emit_plan_table(tasks: List[Dict[str, Any]]) -> str:
    if not tasks:
        return "<p>No implementation items — great job.</p>"
    rows = []
    rows.append("<tr><th>Status</th><th>URL</th><th>Task</th><th>Why/impact</th><th>Effort</th><th>Priority</th><th>Patch</th></tr>")
    for t in tasks:
        rows.append(
            f"<tr><td>{t['status']}</td><td><a href=\"{_h(t['url'])}\">{_h(t['url'])}</a></td>"
            f"<td>{_h(t['task'])}</td><td>{_h(t['why'])}</td>"
            f"<td>{_h(t['effort'])}</td><td>{t['priority']}</td><td>{t['patch']}</td></tr>"
        )
    return "<table class='grid'>" + "\n".join(rows) + "</table>"

# ========= GEO block =========

def _emit_geo_recs(site_url: str) -> str:
    host = _host(site_url)
    brand = host.split(":")[0].split(".")[-2].capitalize() if "." in host else host.capitalize()
    org_jsonld = {
        "@context": "https://schema.org",
        "@type": "Organization",
        "name": brand or "Organization",
        "url": f"https://{host}/" if host else site_url,
        "logo": f"https://{host}/favicon-512.png" if host else "",
        "sameAs": []
    }
    website_jsonld = {
        "@context": "https://schema.org",
        "@type": "WebSite",
        "name": brand or "Website",
        "url": f"https://{host}/" if host else site_url
    }
    block = f"""
<h2>GEO Recommendations (entity/schema)</h2>
<p>Voeg <code>Organization</code> en <code>WebSite</code> JSON-LD toe op de homepage, met een geldige logo-URL en <code>sameAs</code>-profielen (LinkedIn, Wikidata). Versterkt entity grounding voor AI en klassieke SEO.</p>
<pre>{html.escape(json.dumps(org_jsonld, ensure_ascii=False, indent=2))}</pre>
<pre>{html.escape(json.dumps(website_jsonld, ensure_ascii=False, indent=2))}</pre>
"""
    return block

# ========= LLM narrative (with graceful fallback) =========

def _try_llm_generate(prompt: str, system: Optional[str], model: Optional[str], temperature: float, max_tokens: int) -> Optional[str]:
    try:
        import llm  # project-local helper, interface may vary
    except Exception:
        return None
    # strategy 1: llm.generate(prompt, system=..., model=...)
    try:
        if hasattr(llm, "generate"):
            out = llm.generate(prompt, system=system, model=model, temperature=temperature, max_tokens=max_tokens)
            if isinstance(out, dict):
                txt = out.get("text") or out.get("content") or ""
            else:
                txt = str(out)
            return _escape_ascii(txt).strip() or None
    except Exception:
        pass
    # strategy 2: llm.chat([{role,content}...])
    try:
        if hasattr(llm, "chat"):
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
            out = llm.chat(messages=msgs, model=model, temperature=temperature, max_tokens=max_tokens)
            if isinstance(out, dict):
                txt = out.get("content") or out.get("text") or ""
            else:
                txt = str(out)
            return _escape_ascii(txt).strip() or None
    except Exception:
        pass
    # strategy 3: llm.complete(prompt, ...)
    try:
        if hasattr(llm, "complete"):
            out = llm.complete(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
            if isinstance(out, dict):
                txt = out.get("text") or out.get("content") or ""
            else:
                txt = str(out)
            return _escape_ascii(txt).strip() or None
    except Exception:
        pass
    return None

def _compose_llm_prompt_nl(context: Dict[str, Any]) -> str:
    # Compact, anti-hallucinatie: alle cijfers/URL’s worden als JSON meegegeven.
    ctx_json = json.dumps(context, ensure_ascii=False)
    return (
        "Je bent een nuchtere SEO/AEO auditor. Schrijf in het Nederlands een kort, actiegericht rapport voor een CMO.\n"
        "Regels:\n"
        "1) Gebruik uitsluitend feiten uit de meegeleverde JSON, verzin niets. Als iets ontbreekt: noem 'N.v.t.'\n"
        "2) Maximaal 180 woorden per sectie, concrete URL's waar relevant.\n"
        "3) Secties en koppen exact in deze volgorde:\n"
        "   - Executive summary\n"
        "   - Key actions (7 dagen)\n"
        "   - Quarterly outlook\n"
        "JSON:\n"
        f"{ctx_json}\n"
        "Produceer plain text met alinea's en opsommingen; geen Markdown codeblokken."
    )

def _build_llm_context(site_url: str, crawl: Dict[str, Any], crawl_an: Dict[str, Any], aeo_pages: List[Dict[str, Any]], keywords: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    pages = crawl.get("pages") or []
    # top issues
    bad_titles = [f for f in (crawl_an.get("fixes") or []) if f["field"] == "title"]
    bad_descs  = [f for f in (crawl_an.get("fixes") or []) if f["field"] == "meta_description"]
    canon_issues = crawl_an.get("canonical_patches") or []
    og_issues    = crawl_an.get("og_patches") or []
    # aeo summary
    aeo_compact = []
    for p in aeo_pages:
        aeo_compact.append({
            "url": p.get("url"),
            "type": p.get("type"),
            "score": p.get("score"),
            "issues": p.get("issues"),
            "qas_detected": (p.get("metrics") or {}).get("qas_detected")
        })
    # keywords sample
    kw_sample = (keywords or {}).get("keywords") or []
    kw_sample = kw_sample[:10]

    return {
        "site_url": site_url,
        "pages_crawled": len(pages),
        "seo": {
            "titles_needing_work": len(bad_titles),
            "descriptions_needing_work": len(bad_descs),
            "canonical_issues": len(canon_issues),
            "og_issues": len(og_issues),
            "example_title_fix": bad_titles[0] if bad_titles else None,
            "example_desc_fix": bad_descs[0] if bad_descs else None
        },
        "aeo": aeo_compact[:15],
        "keywords_sample": kw_sample
    }

def _render_narrative_html(title: str, text: str, label_class: str) -> str:
    # simpele linebreak → <br>, bullets blijven leesbaar zonder externe md-parser
    safe = _h(text)
    safe = safe.replace("\n", "<br>")
    return f"<h2>{_h(title)}</h2><div class='{label_class}'>{safe}</div>"

def _compose_exec_summary_fallback(site_url: str, crawl: Dict[str, Any], crawl_an: Dict[str, Any], aeo_pages: List[Dict[str, Any]]) -> str:
    pages = crawl.get("pages") or []
    n_pages = len(pages)
    bad_titles = sum(1 for f in (crawl_an.get("fixes") or []) if f["field"] == "title")
    bad_descs  = sum(1 for f in (crawl_an.get("fixes") or []) if f["field"] == "meta_description")
    canon_issues = len(crawl_an.get("canonical_patches") or [])
    og_issues = len(crawl_an.get("og_patches") or [])
    faq_pages = [p for p in aeo_pages if (p.get("type") or "") == "faq"]
    faq_needing_schema = [p for p in faq_pages if any("No FAQPage JSON-LD." in x for x in (p.get("issues") or []))]
    lines = []
    lines.append(f"Scope: {n_pages} pagina's gecrawld op {_h(_host(site_url))}.")
    lines.append(f"SEO: {bad_titles} titels en {bad_descs} meta-descriptions vragen werk; {canon_issues} canonical- en {og_issues} Open Graph-patches voorgesteld.")
    if faq_pages:
        lines.append(f"AEO: {len(faq_pages)} FAQ-pagina('s); {len(faq_needing_schema)} missen FAQPage JSON-LD.")
    else:
        lines.append("AEO: geen FAQ-pagina's gedetecteerd.")
    return " ".join(lines)

def _compose_actions_fallback(crawl_an: Dict[str, Any], aeo_pages: List[Dict[str, Any]]) -> str:
    actions: List[str] = []
    for c in (crawl_an.get("canonical_patches") or [])[:5]:
        actions.append(f"Fix canonical op {_norm_url(c['url'])}.")
    need_desc = [f for f in (crawl_an.get("fixes") or []) if f["field"] == "meta_description"][:5]
    for f in need_desc:
        actions.append(f"Schrijf meta description voor {_norm_url(f['url'])}.")
    faq_schema_miss = [p for p in aeo_pages if (p.get("type") == "faq") and any('No FAQPage JSON-LD.' in x for x in (p.get("issues") or []))][:5]
    for p in faq_schema_miss:
        actions.append(f"Voeg FAQPage JSON-LD toe op {_norm_url(p['url'])}.")
    if not actions:
        actions.append("Geen kritieke acties voor de komende 7 dagen.")
    return "• " + "<br>• ".join(_h(a) for a in actions)

def _compose_quarter_fallback(aeo_pages: List[Dict[str, Any]]) -> str:
    low = sorted(aeo_pages, key=lambda p: (p.get("score") or 0))[:3]
    urls = ", ".join([_norm_url(p.get("url") or "") for p in low]) if low else "N.v.t."
    return f"Focus Q-o-Q op het verhogen van AEO-scores via answer-first content en schema. Begin met: {urls}."

# ========= HTML wrapper =========

def _html_doc(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="nl">
<head>
<meta charset="utf-8">
<title>{_h(title)}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.45;padding:24px;color:#111;background:#fff}}
h1,h2,h3{{margin:0 0 10px}}
h1{{font-size:22px}} h2{{font-size:18px;margin-top:24px}} h3{{font-size:16px;margin-top:16px}}
small.mono,span.mono{{font-family:ui-monospace,Menlo,Consolas,monospace;color:#666}}
table.grid{{border-collapse:collapse;width:100%;margin:8px 0 16px}}
table.grid th,table.grid td{{border:1px solid #ddd;padding:8px;vertical-align:top}}
table.grid th{{background:#f7f7f7;text-align:left}}
pre{{white-space:pre-wrap;word-wrap:break-word;background:#f8f8f8;border:1px solid #eee;padding:8px;border-radius:6px}}
div.narrative{{background:#fff;border:1px solid #eee;padding:12px;border-radius:6px}}
</style>
</head>
<body>
{body}
</body>
</html>"""

# ========= CSV builders =========

def _build_patches_csv(fixes: List[Dict[str, Any]]) -> str:
    if not fixes:
        return "Page URL,Field,Issue,Current,Proposed\n"
    rows = ["Page URL,Field,Issue,Current,Proposed"]
    for f in fixes:
        rows.append(",".join([
            _csv_escape(f["url"]),
            _csv_escape(f["field"]),
            _csv_escape(f["issue"]),
            _csv_escape(f["current"]),
            _csv_escape(f["proposed"]),
        ]))
    return "\n".join(rows) + "\n"

def _build_plan_csv(tasks: List[Dict[str, Any]]) -> str:
    if not tasks:
        return "Status,URL,Task,Why/impact,Effort,Priority,Patch\n"
    rows = ["Status,URL,Task,Why/impact,Effort,Priority,Patch"]
    for t in tasks:
        rows.append(",".join([
            _csv_escape(t["status"]),
            _csv_escape(t["url"]),
            _csv_escape(t["task"]),
            _csv_escape(t["why"]),
            _csv_escape(t["effort"]),
            str(t["priority"]),
            _csv_escape(t["patch"]),
        ]))
    return "\n".join(rows) + "\n"

# ========= Main =========

def generate_report(conn, job):
    site_id = job["site_id"]
    payload = job.get("payload") or {}
    use_llm = bool(payload.get("use_llm") or (os.getenv("REPORT_USE_LLM", "0") == "1"))
    model = payload.get("llm_model") or os.getenv("REPORT_LLM_MODEL") or None
    temperature = float(payload.get("llm_temperature") or os.getenv("REPORT_LLM_TEMPERATURE") or 0.2)
    max_tokens = int(payload.get("llm_max_tokens") or os.getenv("REPORT_LLM_MAX_TOKENS") or 900)

    aeo = _fetch_latest(conn, site_id, "aeo")
    crawl = _fetch_latest(conn, site_id, "crawl")
    keywords = _fetch_latest(conn, site_id, "keywords")

    if not aeo:
        raise ValueError("AEO job missing; run 'aeo' before 'report'.")
    if not crawl:
        raise ValueError("Crawl job missing; run 'crawl' before 'report'.")

    site_url = ((aeo.get("site") or {}).get("url")) or ((crawl.get("start_url") or "")) or ""
    site_url = _norm_url(site_url)
    title = f"SEO • GEO • AEO Audit — {site_url or ''}"

    # AEO sections
    pages_aeo = aeo.get("pages") or []
    aeo_scorecards_html = _emit_aeo_scorecards(pages_aeo)
    aeo_qna_html = _emit_aeo_qna(pages_aeo)

    # SEO analysis
    crawl_analysis = _analyze_seo_from_crawl(crawl)
    fixes = crawl_analysis["fixes"]
    html_patches = _emit_html_patches(crawl_analysis["canonical_patches"], crawl_analysis["og_patches"])
    dup_titles_html = _emit_duplicate_groups(crawl_analysis["dup_titles"], "titles")
    dup_descs_html = _emit_duplicate_groups(crawl_analysis["dup_descs"], "meta descriptions")
    plan_tasks = _build_plan(crawl_analysis, fixes)
    plan_table_html = _emit_plan_table(plan_tasks)

    # Narrative (LLM or fallback)
    narrative_blocks: List[str] = []
    llm_text = None
    if use_llm:
        ctx = _build_llm_context(site_url, crawl, crawl_analysis, pages_aeo, keywords)
        prompt = _compose_llm_prompt_nl(ctx)
        system = "Je bent een senior SEO/AEO consultant. Wees feitelijk, bondig, en data-gedreven."
        llm_text = _try_llm_generate(prompt=prompt, system=system, model=model, temperature=temperature, max_tokens=max_tokens)

    if llm_text:
        narrative_blocks.append(_render_narrative_html("Executive narrative (LLM)", llm_text, "narrative"))
    else:
        # deterministic fallback in 3 secties
        narrative_blocks.append(_render_narrative_html("Executive summary", _compose_exec_summary_fallback(site_url, crawl, crawl_analysis, pages_aeo), "narrative"))
        narrative_blocks.append(_render_narrative_html("Key actions (7 dagen)", _compose_actions_fallback(crawl_analysis, pages_aeo), "narrative"))
        narrative_blocks.append(_render_narrative_html("Quarterly outlook", _compose_quarter_fallback(pages_aeo), "narrative"))

    # Assemble HTML body
    body = []
    body.append(f"<h1>{_h(title)}</h1>")
    body.append(f"<small class='mono'>Generated: {_h(_now_utc_iso())}</small>")
    body.extend(narrative_blocks)

    body.append("<h2>AEO — Answer readiness (scorecards)</h2>")
    body.append(aeo_scorecards_html)
    body.append("<h2>AEO — Q&A (snippet-ready, ready-to-paste)</h2>")
    body.append(aeo_qna_html)

    body.append("<h2>SEO — Concrete text fixes (ready-to-paste)</h2>")
    body.append(_emit_seo_fixes(fixes))

    body.append(html_patches)

    body.append("<h2>Duplicate groups — titles</h2>")
    body.append(dup_titles_html)
    body.append("<h2>Duplicate groups — meta descriptions</h2>")
    body.append(dup_descs_html)

    body.append("<h2>Implementation plan (priority × impact × effort)</h2>")
    body.append(plan_table_html)

    body.append(_emit_geo_recs(site_url))

    # Optional keywords sample
    if keywords and (keywords.get("keywords") or []):
        kws = [str(x) for x in (keywords.get("keywords") or [])][:10]
        body.append("<h2>Keywords — sample</h2><p><span class='mono'>" + _h(", ".join(kws)) + "</span></p>")

    html_doc = _html_doc(title, "\n".join(body))
    html_b64 = base64.b64encode(html_doc.encode("utf-8")).decode("ascii")

    patches_csv = _build_patches_csv(fixes)
    plan_csv = _build_plan_csv(plan_tasks)
    patches_b64 = base64.b64encode(patches_csv.encode("utf-8")).decode("ascii")
    plan_b64 = base64.b64encode(plan_csv.encode("utf-8")).decode("ascii")

    return {
        "html_base64": html_b64,
        "patches_csv_base64": patches_b64,
        "plan_csv_base64": plan_b64
    }
