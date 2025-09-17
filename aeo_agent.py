# aeo_agent.py

import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit
from html import unescape as html_unescape

from psycopg.rows import dict_row

# -----------------------------------------------------
# Config
# -----------------------------------------------------
MAX_ANS_WORDS = 80
MIN_QA_ON_FAQ_PAGE = 3
OPENAI_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # niet gebruikt hier; heuristieken only

# -----------------------------------------------------
# DB helpers
# -----------------------------------------------------
def _fetch_latest_job(conn, site_id: str, jtype: str):
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


def _fetch_site_meta(conn, site_id: str) -> Dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT s.url, s.language, s.country, a.name AS account_name
              FROM sites s
              JOIN accounts a ON a.id = s.account_id
             WHERE s.id=%s
            """,
            (site_id,),
        )
        return cur.fetchone() or {}


# -----------------------------------------------------
# URL / normalize
# -----------------------------------------------------
def _norm_url(u: str) -> str:
    if not u:
        return ""
    p = urlsplit(u)
    scheme = p.scheme or "https"
    host = p.hostname or ""
    path = (p.path or "/")
    if not path:
        path = "/"
    if not path.startswith("/"):
        path = "/" + path
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return urlunsplit((scheme, host, path, "", ""))


def _unique_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bucket: Dict[str, Dict[str, Any]] = {}

    def _score(pp: Dict[str, Any]) -> Tuple[int, int]:
        return (1 if (pp.get("status") == 200) else 0, int(pp.get("word_count") or 0))

    for p in pages or []:
        u = _norm_url(p.get("final_url") or p.get("url") or "")
        if not u:
            continue
        cur = bucket.get(u)
        if (cur is None) or (_score(p) > _score(cur)):
            bucket[u] = p
    return [{**v, "final_url": u, "url": u} for u, v in bucket.items()]


# -----------------------------------------------------
# Language (simple heuristic)
# -----------------------------------------------------
_NL_HINTS = [" de ", " het ", " een ", " en ", " voor ", " met ", " jouw ", " je ", " wij ", " onze "]
_EN_HINTS = [" the ", " and ", " for ", " with ", " your ", " we ", " our ", " to ", " of "]

def _detect_lang(texts: List[str], site_lang: Optional[str]) -> str:
    site = (site_lang or "").lower()
    default = "nl" if site.startswith("nl") else "en"
    sample = (" ".join([t for t in texts if t])[:800] + " ").lower()
    nl_score = sum(1 for w in _NL_HINTS if w in sample)
    en_score = sum(1 for w in _EN_HINTS if w in sample)
    if nl_score > en_score:
        return "nl"
    if en_score > nl_score:
        return "en"
    return default


# -----------------------------------------------------
# Page-type classifier
# -----------------------------------------------------
def _classify_page_type(url: str, title: str, h1: str) -> str:
    u = url.lower()
    path = urlsplit(u).path or "/"
    t = (title or "").lower()
    h = (h1 or "").lower()

    def has(*keys: str) -> bool:
        return any(k in path or k in t or k in h for k in keys)

    if path == "/" or path in ("/index", "/index.html"):
        return "home"
    if has("/faq", " faq", "/veelgestelde-vragen"):
        return "faq"
    return "other"


# -----------------------------------------------------
# Text helpers
# -----------------------------------------------------
_TAG_RE = re.compile(r"<[^>]+>")

def _strip_html(s: str) -> str:
    if not s:
        return ""
    s = html_unescape(s)
    s = _TAG_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _word_count(text: str) -> int:
    return len((text or "").strip().split())


def _trim_words(s: str, limit: int = MAX_ANS_WORDS) -> Tuple[str, bool]:
    words = (s or "").split()
    if len(words) <= limit:
        return " ".join(words), False
    return " ".join(words[:limit]) + "…", True


def _normalize_question(q: str) -> str:
    q = _strip_html(q)
    q = re.sub(r"\s+", " ", q).strip()
    if q and not q.endswith("?"):
        q = q + "?"
    return q


# -----------------------------------------------------
# JSON-LD parsers (robust incl. @graph)
# -----------------------------------------------------
def _qas_from_jsonld(faq_jsonld: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []

    def _handle_entity(entity):
        if not isinstance(entity, dict):
            return
        q = entity.get("name") or entity.get("question") or ""
        ans = entity.get("acceptedAnswer") or entity.get("accepted_answer")
        a_text = ""
        if isinstance(ans, list) and ans:
            ans = ans[0]
        if isinstance(ans, dict):
            a_text = ans.get("text") or ans.get("answer") or ""
        q = _normalize_question(q)
        a_text = _strip_html(a_text)
        if q and a_text:
            out.append({"q": q, "a": a_text})

    def _handle_root(obj):
        if not isinstance(obj, dict):
            return
        if "@graph" in obj and isinstance(obj["@graph"], list):
            for node in obj["@graph"]:
                if isinstance(node, dict) and str(node.get("@type", "")).lower() == "faqpage":
                    main = node.get("mainEntity") or node.get("main_entity") or []
                    if isinstance(main, list):
                        for ent in main:
                            _handle_entity(ent)
                    else:
                        _handle_entity(main)
        main = obj.get("mainEntity") or obj.get("main_entity") or []
        if isinstance(main, list):
            for ent in main:
                _handle_entity(ent)
        else:
            _handle_entity(main)

    if isinstance(faq_jsonld, list):
        for item in faq_jsonld:
            if isinstance(item, dict):
                t = str(item.get("@type", "")).lower()
                if t == "faqpage" or "mainEntity" in item or "@graph" in item:
                    _handle_root(item)
    elif isinstance(faq_jsonld, dict):
        _handle_root(faq_jsonld)

    # Dedup by normalized question
    seen = set()
    deduped: List[Dict[str, str]] = []
    for qa in out:
        key = qa["q"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(qa)
    return deduped


# -----------------------------------------------------
# Visible FAQ extraction (headings/paragraphs)
# -----------------------------------------------------
def _qas_from_visible(page: Dict[str, Any]) -> List[Dict[str, str]]:
    qas: List[Dict[str, str]] = []

    paras: List[str] = page.get("paragraphs") or []
    h2: List[str] = page.get("h2") or []
    h3: List[str] = page.get("h3") or []

    # Strategy 1: pair question-like lines (end with '?') with next paragraph
    for i, p in enumerate(paras):
        p_clean = _strip_html(p)
        if not p_clean:
            continue
        if p_clean.endswith("?") and len(p_clean) <= 200:
            # try answer in next non-empty paragraph
            a = ""
            for j in range(i + 1, min(i + 4, len(paras))):
                nxt = _strip_html(paras[j])
                if nxt:
                    a = nxt
                    break
            if a:
                qas.append({"q": _normalize_question(p_clean), "a": a})

    # Strategy 2: H2/H3 that look like questions -> first paragraph after that heading
    headings = [("h2", h2), ("h3", h3)]
    for tag, lst in headings:
        for i, h in enumerate(lst):
            h_clean = _strip_html(h)
            if not h_clean:
                continue
            if h_clean.endswith("?") and len(h_clean) <= 200:
                # try to map to paragraphs by position if available via a simple heuristic:
                # take the first paragraph that mentions ≥3 words from heading
                cand = ""
                h_words = [w.lower() for w in re.findall(r"\w+", h_clean) if len(w) > 2][:6]
                for p in paras:
                    pw = p.lower()
                    hits = sum(1 for w in h_words if w in pw)
                    if hits >= 3:
                        cand = _strip_html(p)
                        break
                if not cand and paras:
                    cand = _strip_html(paras[min(i, len(paras)-1)])
                if cand:
                    qas.append({"q": _normalize_question(h_clean), "a": cand})

    # Deduplicate
    seen = set()
    out: List[Dict[str, str]] = []
    for qa in qas:
        k = (qa["q"].strip().lower(), qa["a"][:80].strip().lower())
        if k in seen:
            continue
        seen.add(k)
        out.append(qa)
    return out


# -----------------------------------------------------
# QA Review & Improvements
# -----------------------------------------------------
def _review_and_improve_qas(qas: List[Dict[str, str]], lang: str) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]], Dict[str, int]]:
    reviews: List[Dict[str, Any]] = []
    improved: List[Dict[str, str]] = []
    dup_count = 0
    overlong_count = 0
    ok_count = 0

    # Dedup by question text
    seen_q = set()
    for qa in qas:
        q = _normalize_question(qa.get("q") or "")
        a = _strip_html(qa.get("a") or "")
        if not q or not a:
            continue

        issues = []
        status = "ok"
        # Duplicate?
        key = q.strip().lower()
        if key in seen_q:
            dup_count += 1
            issues.append("Duplicate question")
            status = "improve"
        else:
            seen_q.add(key)

        # Question punctuation/length
        if len(q) > 200:
            issues.append("Question too long")
            status = "improve"

        # Answer length
        trimmed, was_trimmed = _trim_words(a, MAX_ANS_WORDS)
        if was_trimmed:
            overlong_count += 1
            issues.append(f"Answer >{MAX_ANS_WORDS} words")
            status = "improve"

        # Suggest improved Q if needed
        q_suggest = q
        if not q.endswith("?"):
            q_suggest = q + "?"

        # Suggest improved A if needed (trim only; keep facts intact)
        a_suggest = trimmed

        reviews.append(
            {
                "status": status,
                "issues": issues or ["OK"],
                "current_q": q,
                "current_a": a,
                "suggested_q": q_suggest,
                "suggested_a": a_suggest,
            }
        )

        improved.append({"q": q_suggest, "a": a_suggest})

        if status == "ok":
            ok_count += 1

    metrics = {
        "qas_total": len(qas),
        "qas_unique": len(seen_q),
        "answers_over_80w": overlong_count,
        "answers_leq_80w": max(0, len(improved) - overlong_count),
        "dup_questions": dup_count,
        "qas_ok": ok_count,
    }
    return improved, reviews, metrics


# -----------------------------------------------------
# Build HTML FAQ block & JSON-LD
# -----------------------------------------------------
def _faq_html_block(qas: List[Dict[str, str]], lang: str) -> str:
    label = "Veelgestelde vragen" if lang == "nl" else "Frequently asked questions"
    lis = []
    for qa in qas:
        q = _normalize_question(qa["q"])
        a = _strip_html(qa["a"])
        lis.append(
            f"""<li class="faq-item">
  <h3 class="faq-q">{q}</h3>
  <p class="faq-a">{a}</p>
</li>"""
        )
    return f"""<section id="faq" aria-labelledby="faq-title">
  <h2 id="faq-title">{label}</h2>
  <ul class="faq-list">
    {''.join(lis)}
  </ul>
</section>"""


def _faq_jsonld(qas: List[Dict[str, str]]) -> Dict[str, Any]:
    items = []
    for qa in qas:
        items.append(
            {
                "@type": "Question",
                "name": _normalize_question(qa["q"]),
                "acceptedAnswer": {"@type": "Answer", "text": _strip_html(qa["a"])},
            }
        )
    return {"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": items}


# -----------------------------------------------------
# Scoring (AEO quality)
# -----------------------------------------------------
def _score_page(ptype: str, has_faq_schema: bool, qas_n: int, answers_leq_80w: int, dup_q: int, parity_ok: Optional[bool]) -> Tuple[int, List[str]]:
    issues: List[str] = []
    score = 0

    if ptype == "faq":
        score = 40
        if qas_n >= MIN_QA_ON_FAQ_PAGE:
            score += 20
        else:
            issues.append("Too few Q&A (min 3).")

        if has_faq_schema:
            score += 20
        else:
            issues.append("No FAQPage JSON-LD.")

        if qas_n > 0:
            frac = answers_leq_80w / max(1, qas_n)
            score += int(15 * frac)
            if answers_leq_80w < qas_n:
                issues.append("Overlong answers present.")
        else:
            issues.append("No Q&A detected.")

        if dup_q > 0:
            issues.append("Duplicate questions present.")

        if parity_ok is False:
            issues.append("Schema/text parity mismatch.")
        score = max(0, min(100, score))
    else:
        # Non-FAQ pages: light signal (presence of micro-FAQ boosts a bit)
        score = 50 if (qas_n >= 3 and has_faq_schema) else (30 if qas_n >= 1 else 15)
        if qas_n == 0:
            issues.append("No Q&A section on page.")
        score = max(0, min(100, score))

    return score, issues


# -----------------------------------------------------
# Parity check (visible vs JSON-LD)
# -----------------------------------------------------
def _parity_ok(visible: List[Dict[str, str]], jsonld_qas: List[Dict[str, str]]) -> Optional[bool]:
    if not jsonld_qas:
        return None
    if not visible:
        # JSON-LD exists but no visible Q&A — usually not ideal
        return False
    vis_qs = {qa["q"].strip().lower() for qa in visible}
    ld_qs = {qa["q"].strip().lower() for qa in jsonld_qas}
    # Parity = largely overlapping
    inter = vis_qs.intersection(ld_qs)
    ratio = len(inter) / max(1, len(vis_qs))
    return ratio >= 0.6


# -----------------------------------------------------
# Main generator
# -----------------------------------------------------
def generate_aeo(conn, job):
    site_id = job["site_id"]
    site_meta = _fetch_site_meta(conn, site_id)
    crawl = _fetch_latest_job(conn, site_id, "crawl")

    pages_out: List[Dict[str, Any]] = []

    if not crawl or not crawl.get("pages"):
        return {"pages": pages_out}

    pages = _unique_pages(crawl.get("pages") or [])

    for p in pages:
        url = _norm_url(p.get("final_url") or p.get("url") or "")
        title = (p.get("title") or "").strip()
        h1 = (p.get("h1") or "").strip()
        jsonld_types = [t.lower() for t in (p.get("jsonld_types") or [])]
        page_lang = _detect_lang([title, h1] + (p.get("paragraphs") or []), site_meta.get("language"))
        ptype = _classify_page_type(url, title, h1)

        # Extract visible Q&A
        qas_vis = _qas_from_visible(p)

        # Extract Q&A from JSON-LD if crawl captured any FAQ structure content on the page (rare)
        # The crawl payload may not include raw JSON-LD; we therefore rely on visible + create our own improved JSON-LD.
        qas_ld_raw = []
        # If the crawler stored some faq-like structure under p.get("faq_jsonld_raw"), parse it:
        raw_jld = p.get("faq_jsonld") or p.get("jsonld") or p.get("json_ld") or None
        if raw_jld:
            try:
                qas_ld_raw = _qas_from_jsonld(raw_jld)
            except Exception:
                qas_ld_raw = []

        # Merge (prefer visible), dedupe by question
        merged_map: Dict[str, Dict[str, str]] = {}
        for src in (qas_vis, qas_ld_raw):
            for qa in src:
                key = (_normalize_question(qa["q"]).strip().lower())
                if key not in merged_map:
                    merged_map[key] = {"q": _normalize_question(qa["q"]), "a": _strip_html(qa["a"])}

        merged_qas = list(merged_map.values())

        # Review + improve (trim overlong to ≤80 words)
        improved_qas, reviews, qa_metrics = _review_and_improve_qas(merged_qas, page_lang)

        has_faq_schema = any(t == "faqpage" for t in jsonld_types)
        parity = _parity_ok(merged_qas, qas_ld_raw)

        # Score + page issues
        score, issues = _score_page(
            ptype=ptype,
            has_faq_schema=has_faq_schema,
            qas_n=len(improved_qas),
            answers_leq_80w=qa_metrics["answers_leq_80w"],
            dup_q=qa_metrics["dup_questions"],
            parity_ok=parity,
        )

        # Build patches (only when there are Q&A or page is FAQ)
        content_patches: List[Dict[str, Any]] = []
        if ptype == "faq" or improved_qas:
            # HTML block (improved, ≤80w)
            faq_html = _faq_html_block(improved_qas, page_lang)
            content_patches.append({
                "url": url,
                "field": "faq_html_block",
                "category": "body",
                "problem": "Improve FAQ block (trim to ≤80 words; dedupe; clear questions)" if merged_qas else "Add FAQ block (3–6 Q&A ≤80w)",
                "current": "(existing FAQ on page)" if merged_qas else "(none)",
                "proposed": "Replace or add FAQ section with cleaned Q&A.",
                "html_patch": faq_html,
                "severity": 2,
                "impact": 5,
                "effort": 2,
                "priority": 6.0,
                "patchable": True,
            })

            # JSON-LD (built from improved Q&A)
            faq_ld = _faq_jsonld(improved_qas)
            content_patches.append({
                "url": url,
                "field": "faq_jsonld",
                "category": "head",
                "problem": "Add/refresh FAQPage JSON-LD to match visible Q&A.",
                "current": "(present but may not match)" if has_faq_schema else "(none)",
                "proposed": "Inject <script type='application/ld+json'>…</script> mirroring visible FAQ.",
                "html_patch": "<script type=\"application/ld+json\">" + json.dumps(faq_ld, ensure_ascii=False) + "</script>",
                "severity": 2,
                "impact": 4,
                "effort": 1,
                "priority": 6.0,
                "patchable": True,
            })
        else:
            faq_ld = {}

        # Per-QA “what’s good / what to fix”
        # Verpak als losse patches wanneer er concrete verbeteringen zijn
        for r in reviews:
            if r["status"] == "improve":
                # Eén patch per QA met current vs proposed
                html_patch = f"<h3>{r['suggested_q']}</h3>\n<p>{r['suggested_a']}</p>"
                content_patches.append({
                    "url": url,
                    "field": "faq_item",
                    "category": "body",
                    "problem": "; ".join(r.get("issues", [])) or "Improve FAQ item",
                    "current": f"Q: {r['current_q']}\nA: {r['current_a']}",
                    "proposed": f"Q: {r['suggested_q']}\nA: {r['suggested_a']}",
                    "html_patch": html_patch,
                    "severity": 1,
                    "impact": 4,
                    "effort": 1,
                    "priority": 5.0,
                    "patchable": True,
                })

        pages_out.append(
            {
                "url": url,
                "lang": page_lang,
                "type": ptype,
                "qas": improved_qas,            # deze worden door report_agent getoond (≤80w)
                "qas_review": reviews,          # detail per Q/A: OK of verbeteren, inclusief current vs proposed
                "faq_jsonld": _faq_jsonld(improved_qas) if (ptype == "faq" or improved_qas) else {},
                "content_patches": content_patches,
                "score": score,
                "issues": issues,
                "metrics": {
                    "qas": len(improved_qas),
                    "has_faq_schema": 1 if has_faq_schema else 0,
                    "answers_leq_80w": qa_metrics["answers_leq_80w"],
                    "answers_gt_80w": qa_metrics["answers_over_80w"],
                    "dup_questions": qa_metrics["dup_questions"],
                    "parity_ok": parity if parity is not None else "",
                },
            }
        )

    return {"pages": pages_out}
