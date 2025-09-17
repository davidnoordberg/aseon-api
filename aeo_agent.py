# aeo_agent.py — AEO (Answer Engine Optimization) analyzer with FAQ gating,
# improved answers, parity checks, near-dup clustering, and scoring.
# Output schema (top-level):
# {
#   "site": {"url": "..."},
#   "pages": [
#     {
#       "url": "...",
#       "type": "faq" | "other",
#       "score": 0..100,
#       "issues": ["..."],
#       "metrics": {
#         "qas_detected": int,
#         "answers_leq_80w": int,
#         "answers_gt_80w": int,
#         "parity_ok": bool,
#         "src_counts": {"jsonld": int, "visible": int, "faq_job": int},
#         "dup_questions": int,
#         "has_faq_schema_detected": bool
#       },
#       "qas": [{"q": "...", "a": "..."}]
#     }
#   ]
# }

import base64
import html
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from psycopg.rows import dict_row

# ============= Utilities =============

def _escape_ascii(s: str) -> str:
    if s is None:
        return ""
    t = html.unescape(str(s))
    repl = {
        "\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00A0": " ",
        "\u200b": "", "\u200d": "", "\ufeff": ""
    }
    for k, v in repl.items():
        t = t.replace(k, v)
    # common mojibake sequences
    t = t.replace("â", "'").replace("â", '"').replace("â", '"')
    t = t.replace("â", "-").replace("â", "-")
    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _norm(s: str) -> str:
    return _escape_ascii(s).lower()

def _h(s: str) -> str:
    return html.escape(_escape_ascii(s or ""))

def _host(u: str) -> str:
    try:
        return urlparse(u).netloc.lower()
    except Exception:
        return ""

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

def _fetch_latest(conn, site_id: str, jtype: str) -> Optional[Dict[str, Any]]:
    from psycopg.rows import dict_row
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

def _has_mojibake(s: str) -> bool:
    return "â" in s or "\ufffd" in s

# ============= JSON-LD & Visible extraction =============

def _extract_faq_from_jsonld(page: Dict[str, Any]) -> List[Dict[str, str]]:
    qas: List[Dict[str, str]] = []
    jsonlds = page.get("jsonld") or page.get("ld_json") or []
    if isinstance(jsonlds, dict):
        jsonlds = [jsonlds]
    for j in jsonlds:
        if not isinstance(j, dict):
            continue
        t = j.get("@type")
        t_list = [t] if isinstance(t, str) else (t or [])
        if any(x.lower() == "faqpage" for x in [str(xx) for xx in t_list]):
            ents = j.get("mainEntity") or []
            if isinstance(ents, dict):
                ents = [ents]
            for e in ents:
                name = _escape_ascii(str(e.get("name") or e.get("question") or ""))
                ans = ""
                aa = e.get("acceptedAnswer") or {}
                if isinstance(aa, list):
                    aa = aa[0] if aa else {}
                ans = _escape_ascii(str(aa.get("text") or aa.get("answer") or ""))
                if name and ans:
                    qas.append({"q": name, "a": ans, "source": "jsonld"})
    return qas

def _extract_faq_from_visible(page: Dict[str, Any]) -> List[Dict[str, str]]:
    # Prefer explicit fields produced by crawler, else light heuristics
    candidates = []
    for key in ["visible_faq", "visible_qas", "faq_visible", "visibleFAQ", "visible"]:
        v = page.get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict) and "q" in v[0] and "a" in v[0]:
            candidates = v
            break
    qas: List[Dict[str, str]] = []
    if candidates:
        for item in candidates:
            q = _escape_ascii(str(item.get("q","")))
            a = _escape_ascii(str(item.get("a","")))
            if q and a:
                qas.append({"q": q, "a": a, "source": "visible"})
        return qas

    # Heuristic fallback: scan headings + paragraphs block (very conservative)
    heads = (page.get("h2") or []) + (page.get("h3") or [])
    pars = page.get("paragraphs") or []
    i = 0
    while i < len(heads):
        q = _escape_ascii(str(heads[i] or ""))
        if not q:
            i += 1
            continue
        # find the next paragraph as answer
        a = ""
        for p in pars:
            if len(_escape_ascii(p)) > 20:
                a = _escape_ascii(p)
                break
        if q and a:
            qas.append({"q": q, "a": a, "source": "visible"})
        i += 1
    return qas

def _is_faq_page(url: str, page: Dict[str, Any], schema_qas: List[Dict[str,str]], visible_qas: List[Dict[str,str]]) -> bool:
    if "/faq" in url.lower():
        return True
    if schema_qas and len(schema_qas) >= 3:
        return True
    h2 = [str(x).lower() for x in (page.get("h2") or [])]
    h3 = [str(x).lower() for x in (page.get("h3") or [])]
    if any("faq" in x for x in h2 + h3):
        return True
    if visible_qas and len(visible_qas) >= 3:
        return True
    return False

# ============= Dedup & parity =============

_STOP = set("the a an to for and or but with in on at from of by as is are was were be been being this that it its your our their you we they i me my mine".split())

def _tokenize(s: str) -> List[str]:
    s = _norm(s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t and t not in _STOP]
    return toks

def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0

def _cluster_near_dups(qas: List[Dict[str,str]], threshold: float) -> Tuple[List[List[int]], List[int]]:
    # returns clusters (list of lists of indices) and canonical index per cluster
    n = len(qas)
    used = [False]*n
    clusters: List[List[int]] = []
    canon: List[int] = []
    for i in range(n):
        if used[i]:
            continue
        base = _tokenize(qas[i]["q"])
        group = [i]
        used[i] = True
        for j in range(i+1, n):
            if used[j]:
                continue
            sim = _jaccard(base, _tokenize(qas[j]["q"]))
            if sim >= threshold:
                group.append(j)
                used[j] = True
        clusters.append(group)
        canon.append(group[0])
    return clusters, canon

def _parity(visible_qas: List[Dict[str,str]], schema_qas: List[Dict[str,str]]) -> Tuple[bool, List[str], List[str]]:
    vis_set = set(_norm(x["q"]) for x in visible_qas)
    sch_set = set(_norm(x["q"]) for x in schema_qas)
    missing_in_schema = [x for x in vis_set if x not in sch_set]
    missing_in_visible = [x for x in sch_set if x not in vis_set]
    ok = not missing_in_schema and not missing_in_visible
    return ok, missing_in_schema, missing_in_visible

# ============= Answer improvement =============

def _first_sentence(s: str) -> str:
    s = _escape_ascii(s)
    # split on sentence-ish boundaries
    parts = re.split(r"(?<=[\.\!\?\:])\s+", s)
    return parts[0] if parts else s

def _strip_question_restate(q: str, a: str) -> str:
    qn = _norm(q)
    an = _norm(a)
    if qn and an.startswith(qn[: max(10, len(qn)//2)]):
        # if answer begins by repeating the question, drop that prefix
        idx = len(q)
        return _escape_ascii(a[idx:]).lstrip("-—:,. ").strip()
    return a

def _trim_words(s: str, max_words: int) -> str:
    words = _escape_ascii(s).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])

def _deterministic_rewrite(q: str, a: str, max_words: int) -> str:
    a2 = _escape_ascii(a)
    a2 = _strip_question_restate(q, a2)
    if _has_mojibake(a2):
        a2 = _escape_ascii(a2)
    # prefer first sentence; if too short, keep two sentences
    parts = re.split(r"(?<=[\.\!\?\:])\s+", a2)
    if parts:
        head = parts[0]
        if len(head.split()) < 6 and len(parts) > 1:
            head = (head + " " + parts[1]).strip()
        a2 = head
    a2 = _trim_words(a2, max_words)
    return a2

def _llm_rewrite(q: str, a: str, max_words: int) -> Optional[str]:
    try:
        import llm  # project-local interface
    except Exception:
        return None
    prompt = (
        "Rewrite the answer below in Dutch. Rules: ≤{maxw} words, lead with the answer, no question restatement, no marketing fluff.\n"
        "Question: {q}\n"
        "Answer: {a}\n"
        "Rewrite:"
    ).format(maxw=max_words, q=q, a=a)
    try:
        if hasattr(llm, "generate"):
            out = llm.generate(prompt, system="You are a concise editor.", model=os.getenv("AEO_LLM_MODEL") or None, temperature=float(os.getenv("AEO_LLM_TEMPERATURE") or 0.1), max_tokens=220)
            txt = out.get("text") if isinstance(out, dict) else str(out)
            return _escape_ascii(txt or "").strip() or None
        if hasattr(llm, "complete"):
            out = llm.complete(prompt, model=os.getenv("AEO_LLM_MODEL") or None, temperature=float(os.getenv("AEO_LLM_TEMPERATURE") or 0.1), max_tokens=220)
            txt = out.get("text") if isinstance(out, dict) else str(out)
            return _escape_ascii(txt or "").strip() or None
    except Exception:
        return None
    return None

# ============= Scoring & issues =============

def _score_and_issues(page_type: str,
                      has_schema: bool,
                      n_qas: int,
                      overlong: int,
                      dup_clusters: int,
                      parity_ok: bool) -> Tuple[int, List[str]]:
    issues: List[str] = []
    if page_type != "faq":
        if n_qas == 0:
            return 15, ["No Q&A section on page."]
        # non-faq page but has Q&A snippets (e.g., micro-FAQ sections): treat softly
        base = 95
        if overlong > 0:
            issues.append("Overlong answers (>80 words).")
            base -= 5
        return max(15, min(100, base)), issues

    score = 100
    if n_qas < 3:
        issues.append("Too few Q&A (min 3).")
        score -= 20
    if overlong > 0:
        issues.append("Overlong answers (>80 words).")
        score -= 8
    if dup_clusters > 0:
        issues.append("Duplicate questions.")
        score -= 7
    if not has_schema:
        issues.append("No FAQPage JSON-LD.")
        score -= 15
    if not parity_ok and has_schema:
        issues.append("Schema/text parity mismatch.")
        score -= 8
    score = max(0, min(100, score))
    if n_qas == 0:
        score = 15
    return score, issues

# ============= Main analyzer =============

def _analyze_page(page: Dict[str, Any],
                  url: str,
                  only_faq: bool,
                  emit_improved: bool,
                  rewrite_mode: str,
                  max_words: int,
                  dedup_threshold: float) -> Optional[Dict[str, Any]]:

    schema_qas = _extract_faq_from_jsonld(page)
    visible_qas = _extract_faq_from_visible(page)

    has_schema = bool(schema_qas)
    page_type = "faq" if _is_faq_page(url, page, schema_qas, visible_qas) else "other"

    if only_faq and page_type != "faq":
        # Still record minimal metrics for non-FAQ? Return a lightweight entry
        n = len(visible_qas)
        if n == 0:
            return {
                "url": url,
                "type": "other",
                "score": 15,
                "issues": ["No Q&A section on page."],
                "metrics": {
                    "qas_detected": 0,
                    "answers_leq_80w": 0,
                    "answers_gt_80w": 0,
                    "parity_ok": True,
                    "src_counts": {"jsonld": len(schema_qas), "visible": len(visible_qas), "faq_job": 0},
                    "dup_questions": 0,
                    "has_faq_schema_detected": has_schema
                },
                "qas": []
            }
        # has incidental Q&As on non-FAQ
        overlong = sum(1 for qa in visible_qas if len(_escape_ascii(qa.get("a","")).split()) > max_words)
        score, issues = _score_and_issues("other", has_schema, n, overlong, 0, True)
        qas_out = []
        for qa in visible_qas:
            a0 = qa.get("a","")
            a_imp = _deterministic_rewrite(qa.get("q",""), a0, max_words) if emit_improved else a0
            qas_out.append({"q": qa.get("q",""), "a": a_imp})
        return {
            "url": url,
            "type": "other",
            "score": score,
            "issues": issues,
            "metrics": {
                "qas_detected": n,
                "answers_leq_80w": sum(1 for qa in qas_out if len(_escape_ascii(qa["a"]).split()) <= max_words),
                "answers_gt_80w": sum(1 for qa in qas_out if len(_escape_ascii(qa["a"]).split()) > max_words),
                "parity_ok": True,
                "src_counts": {"jsonld": len(schema_qas), "visible": len(visible_qas), "faq_job": 0},
                "dup_questions": 0,
                "has_faq_schema_detected": has_schema
            },
            "qas": qas_out
        }

    # For FAQ pages, merge sources (prefer visible text as baseline; fallback to schema)
    merged: List[Dict[str,str]] = []
    for qa in visible_qas:
        merged.append({"q": qa["q"], "a": qa["a"], "src": "visible"})
    # add schema Qs that are not already present
    vis_norm = set(_norm(qa["q"]) for qa in visible_qas)
    for qa in schema_qas:
        if _norm(qa["q"]) not in vis_norm:
            merged.append({"q": qa["q"], "a": qa["a"], "src": "jsonld"})

    # Dedup clustering on questions
    clusters, canon = _cluster_near_dups(merged, threshold=dedup_threshold)
    canonical_qas: List[Dict[str,str]] = []
    for group, ci in zip(clusters, canon):
        base = merged[ci]
        canonical_qas.append(dict(base))

    # Parity check on original (not deduped) sets
    parity_ok, miss_in_schema, miss_in_visible = _parity(visible_qas, schema_qas)

    # Improve answers
    improved_qas: List[Dict[str,str]] = []
    for qa in canonical_qas:
        q = qa["q"]
        a = qa["a"]
        proposed = a
        if emit_improved:
            if rewrite_mode == "llm":
                proposed = _llm_rewrite(q, a, max_words) or _deterministic_rewrite(q, a, max_words)
            else:
                proposed = _deterministic_rewrite(q, a, max_words)
        improved_qas.append({"q": q, "a": proposed})

    # Metrics & issues
    n_qas = len(improved_qas)
    overlong = sum(1 for qa in improved_qas if len(_escape_ascii(qa["a"]).split()) > max_words)
    dup_count = sum(max(0, len(g)-1) for g in clusters)
    score, issues = _score_and_issues("faq", has_schema, n_qas, overlong, dup_count, parity_ok)

    # Extra quality flags can hint issues (not yet added to 'issues' to avoid noise)
    flags = []
    if any(_has_mojibake(qa["a"]) or _has_mojibake(qa["q"]) for qa in merged):
        flags.append("mojibake")
    if any(qa["a"].count("?") >= 2 for qa in merged):
        flags.append("heading_bleed")
    if any(len(_escape_ascii(qa["a"]).split()) < 3 for qa in merged):
        flags.append("vague")

    m = {
        "qas_detected": n_qas,
        "answers_leq_80w": n_qas - overlong,
        "answers_gt_80w": overlong,
        "parity_ok": parity_ok,
        "src_counts": {"jsonld": len(schema_qas), "visible": len(visible_qas), "faq_job": 0},
        "dup_questions": dup_count,
        "has_faq_schema_detected": has_schema,
        "missing_in_schema": miss_in_schema,
        "missing_in_visible": miss_in_visible,
        "flags": flags,
    }

    return {
        "url": url,
        "type": "faq",
        "score": score,
        "issues": issues,
        "metrics": m,
        "qas": improved_qas
    }

# ============= Public entrypoint =============

def generate_aeo(conn, job):
    site_id = job["site_id"]
    payload = job.get("payload") or {}
    only_faq = bool(payload.get("only_faq", True))
    emit_improved = bool(payload.get("emit_improved", True))
    rewrite_mode = str(payload.get("rewrite", "deterministic")).lower()
    max_words = int(payload.get("max_answer_words", 80))
    dedup_threshold = float(payload.get("dedup_threshold", 0.88))

    crawl = _fetch_latest(conn, site_id, "crawl")
    if not crawl:
        raise ValueError("Crawl job missing; run 'crawl' before 'aeo'.")

    start_url = crawl.get("start_url") or crawl.get("site_url") or ""
    pages = crawl.get("pages") or []

    out_pages: List[Dict[str, Any]] = []
    for p in pages:
        url = _norm_url(p.get("url") or "")
        if not url:
            continue
        analyzed = _analyze_page(
            page=p,
            url=url,
            only_faq=only_faq,
            emit_improved=emit_improved,
            rewrite_mode=rewrite_mode,
            max_words=max_words,
            dedup_threshold=dedup_threshold
        )
        if analyzed:
            out_pages.append(analyzed)

    return {
        "site": {"url": _norm_url(start_url or (pages[0].get("url") if pages else ""))},
        "pages": out_pages
    }
