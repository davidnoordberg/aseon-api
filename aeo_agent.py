# aeo_agent.py

import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit
from html import unescape as html_unescape
from psycopg.rows import dict_row

MAX_ANS_WORDS = 80
MIN_QA_ON_FAQ_PAGE = 3

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

def _norm_url(u: str) -> str:
    if not u:
        return ""
    p = urlsplit(u)
    scheme = p.scheme or "https"
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = (p.path or "/")
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

def _classify_page_type(url: str, title: str, h1: str) -> str:
    u = url.lower()
    path = urlsplit(u).path or "/"
    t = (title or "").lower()
    h = (h1 or "").lower()
    def has(*keys: str) -> bool:
        return any(k in path or k in t or k in h for k in keys)
    if has("/faq", " faq", "/veelgestelde-vragen"):
        return "faq"
    return "other"

_TAG_RE = re.compile(r"<[^>]+>")

def _strip_html(s: str) -> str:
    if not s:
        return ""
    s = html_unescape(s)
    s = _TAG_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

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
    seen = set()
    deduped: List[Dict[str, str]] = []
    for qa in out:
        key = qa["q"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(qa)
    return deduped

def _qas_from_visible(page: Dict[str, Any]) -> List[Dict[str, str]]:
    qas: List[Dict[str, str]] = []
    paras: List[str] = page.get("paragraphs") or []
    for i, p in enumerate(paras):
        q = _strip_html(p)
        if not q or len(q) > 200 or not q.endswith("?"):
            continue
        a = ""
        for j in range(i + 1, min(i + 4, len(paras))):
            cand = _strip_html(paras[j])
            if not cand or cand.endswith("?") or len(cand.split()) < 6:
                continue
            a = cand
            break
        if a:
            qas.append({"q": _normalize_question(q), "a": a})
    seen = set()
    out: List[Dict[str, str]] = []
    for qa in qas:
        k = (qa["q"].strip().lower(), qa["a"][:80].strip().lower())
        if k in seen:
            continue
        seen.add(k)
        out.append(qa)
    return out

def _review_and_improve_qas(qas: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]], Dict[str, int]]:
    reviews: List[Dict[str, Any]] = []
    improved: List[Dict[str, str]] = []
    seen_q = set()
    seen_a = set()
    dup_q = 0
    dup_a = 0
    over = 0
    ok = 0
    for qa in qas:
        q = _normalize_question(qa.get("q") or "")
        a = _strip_html(qa.get("a") or "")
        if not q or not a:
            continue
        status = "ok"
        issues = []
        kq = q.strip().lower()
        if kq in seen_q:
            dup_q += 1
            status = "improve"
            issues.append("Duplicate question")
        else:
            seen_q.add(kq)
        trimmed, was_trimmed = _trim_words(a, MAX_ANS_WORDS)
        if was_trimmed:
            over += 1
            status = "improve"
            issues.append(f"Answer >{MAX_ANS_WORDS} words")
        ka = trimmed.strip().lower()
        if ka in seen_a:
            dup_a += 1
            status = "improve"
            issues.append("Duplicate answer text")
        else:
            seen_a.add(ka)
        reviews.append({
            "status": status,
            "issues": issues or ["OK"],
            "current_q": q,
            "current_a": a,
            "suggested_q": q,
            "suggested_a": trimmed
        })
        improved.append({"q": q, "a": trimmed})
        if status == "ok":
            ok += 1
    metrics = {
        "qas_total": len(qas),
        "qas_unique": len(seen_q),
        "answers_over_80w": over,
        "answers_leq_80w": max(0, len(improved) - over),
        "dup_questions": dup_q,
        "dup_answers": dup_a,
        "qas_ok": ok
    }
    return improved, reviews, metrics

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

def _score_page(ptype: str, has_faq_schema: bool, qas_n: int, answers_leq_80w: int, dup_q: int, dup_a: int, parity_ok: Optional[bool]) -> Tuple[int, List[str]]:
    issues: List[str] = []
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
        if dup_a > 0:
            issues.append("Duplicate answers present.")
        if parity_ok is False:
            issues.append("Schema/text parity mismatch.")
        score = max(0, min(100, score))
    else:
        score = 15 if qas_n == 0 else 30
        if qas_n == 0:
            issues.append("No Q&A section on page.")
    return score, issues

def _parity_ok(visible: List[Dict[str, str]], jsonld_qas: List[Dict[str, str]]) -> Optional[bool]:
    if not jsonld_qas:
        return None
    if not visible:
        return False
    vis_qs = {qa["q"].strip().lower() for qa in visible}
    ld_qs = {qa["q"].strip().lower() for qa in jsonld_qas}
    inter = vis_qs.intersection(ld_qs)
    ratio = len(inter) / max(1, len(vis_qs))
    return ratio >= 0.6

def _index_faq_job(faq_job: Optional[Dict[str, Any]]) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, bool]]:
    by_url: Dict[str, List[Dict[str, str]]] = {}
    schema_flags: Dict[str, bool] = {}
    if not faq_job:
        return by_url, schema_flags
    for item in (faq_job.get("faqs") or []):
        src = item.get("url") or item.get("page") or item.get("source") or ""
        u = _norm_url(src)
        if not u:
            continue
        q = _normalize_question(item.get("q") or item.get("question") or "")
        a = _strip_html(item.get("a") or item.get("answer") or "")
        if q and a:
            by_url.setdefault(u, []).append({"q": q, "a": a})
        if isinstance(item.get("has_faq_schema"), bool):
            schema_flags[u] = schema_flags.get(u, False) or bool(item["has_faq_schema"])
    return by_url, schema_flags

def generate_aeo(conn, job):
    site_id = job["site_id"]
    site_meta = _fetch_site_meta(conn, site_id)
    crawl = _fetch_latest_job(conn, site_id, "crawl")
    faq_job = _fetch_latest_job(conn, site_id, "faq")

    pages_out: List[Dict[str, Any]] = []
    if not crawl or not crawl.get("pages"):
        return {"pages": pages_out}

    faq_index, faq_schema_flags = _index_faq_job(faq_job)
    pages = _unique_pages(crawl.get("pages") or [])

    for p in pages:
        url = _norm_url(p.get("final_url") or p.get("url") or "")
        title = (p.get("title") or "").strip()
        h1 = (p.get("h1") or "").strip()
        page_lang = _detect_lang([title, h1] + (p.get("paragraphs") or []), site_meta.get("language"))
        ptype = _classify_page_type(url, title, h1)
        jsonld_types = [t.lower() for t in (p.get("jsonld_types") or [])]
        has_faq_schema_crawl = any(t == "faqpage" for t in jsonld_types)
        qas_visible = _qas_from_visible(p)
        qas_from_faqjob = faq_index.get(url, [])
        has_faq_schema_any = bool(has_faq_schema_crawl or faq_schema_flags.get(url, False))

        merged_map: Dict[str, Dict[str, str]] = {}
        for src in (qas_from_faqjob or qas_visible):
            for qa in src:
                key = (_normalize_question(qa["q"]).strip().lower())
                if key not in merged_map:
                    merged_map[key] = {"q": _normalize_question(qa["q"]), "a": _strip_html(qa["a"])}

        merged_qas = list(merged_map.values())
        improved_qas, reviews, qa_metrics = _review_and_improve_qas(merged_qas)
        parity = _parity_ok(qas_visible, qas_from_faqjob) if qas_from_faqjob else None

        score, issues = _score_page(
            ptype=ptype,
            has_faq_schema=has_faq_schema_any,
            qas_n=len(improved_qas),
            answers_leq_80w=qa_metrics["answers_leq_80w"],
            dup_q=qa_metrics["dup_questions"],
            dup_a=qa_metrics["dup_answers"],
            parity_ok=parity,
        )

        content_patches: List[Dict[str, Any]] = []
        qas_for_output = []
        faq_ld_obj: Dict[str, Any] = {}

        if ptype == "faq":
            qas_for_output = improved_qas
            if improved_qas:
                html_block = _faq_html_block(improved_qas, page_lang)
                faq_ld_obj = _faq_jsonld(improved_qas)
                content_patches.append({
                    "url": url,
                    "field": "faq_html_block",
                    "category": "body",
                    "problem": "Improve/align FAQ block (≤80w; dedupe; parity)",
                    "current": "(existing or partial FAQ)",
                    "proposed": "Replace or align with cleaned FAQ Q&A.",
                    "html_patch": html_block,
                    "severity": 2,
                    "impact": 5,
                    "effort": 2,
                    "priority": 6.0,
                    "patchable": True,
                })
                content_patches.append({
                    "url": url,
                    "field": "faq_jsonld",
                    "category": "head",
                    "problem": "Add/refresh FAQPage JSON-LD to mirror visible FAQ.",
                    "current": "(present or missing)",
                    "proposed": "Inject <script type='application/ld+json'>…</script> matching on-page Q&A.",
                    "html_patch": "<script type=\"application/ld+json\">" + json.dumps(faq_ld_obj, ensure_ascii=False) + "</script>",
                    "severity": 2,
                    "impact": 4,
                    "effort": 1,
                    "priority": 6.0,
                    "patchable": True,
                })

            for r in reviews:
                if r["status"] == "improve":
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
                "qas": qas_for_output,
                "qas_review": reviews if ptype == "faq" else [],
                "faq_jsonld": faq_ld_obj if ptype == "faq" else {},
                "content_patches": content_patches,
                "score": score,
                "issues": issues if issues else ["OK"],
                "metrics": {
                    "qas": len(improved_qas),
                    "has_faq_schema": 1 if has_faq_schema_any else 0,
                    "answers_leq_80w": qa_metrics.get("answers_leq_80w", 0),
                    "answers_gt_80w": qa_metrics.get("answers_over_80w", 0),
                    "dup_questions": qa_metrics.get("dup_questions", 0),
                    "dup_answers": qa_metrics.get("dup_answers", 0),
                    "qas_detected": len(improved_qas),
                    "has_faq_schema_detected": True if has_faq_schema_any else False,
                    "parity_ok": parity if parity is not None else "",
                },
            }
        )

    return {"pages": pages_out}
