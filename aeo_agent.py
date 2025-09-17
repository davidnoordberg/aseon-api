# aeo_agent.py — AEO analyzer with FAQ gating, improved answers, parity checks,
# near-dup clustering, scoring, and strict pre/post-clean for snippet quality.
#
# Payload options:
#   only_faq: bool (default True)
#   emit_improved: bool (default True)
#   rewrite: "deterministic" | "llm" (default "deterministic")
#   max_answer_words: int (default 80)
#   dedup_threshold: float (default 0.88)
#   faq_paths: [str,...]  # optional substrings/paths to treat as FAQ pages (e.g. ["/faq"])
#
# Output (top-level):
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
#         "has_faq_schema_detected": bool,
#         "missing_in_schema": [...],
#         "missing_in_visible": [...],
#         "flags": ["mojibake","heading_bleed","vague", ...]
#       },
#       "qas": [{"q":"...","a":"..."}]
#     }
#   ]
# }

import html
import json
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
        if any(str(xx).lower() == "faqpage" for xx in t_list):
            ents = j.get("mainEntity") or []
            if isinstance(ents, dict):
                ents = [ents]
            for e in ents:
                name = _escape_ascii(str(e.get("name") or e.get("question") or ""))
                aa = e.get("acceptedAnswer") or {}
                if isinstance(aa, list):
                    aa = aa[0] if aa else {}
                ans = _escape_ascii(str(aa.get("text") or aa.get("answer") or ""))
                if name and ans:
                    qas.append({"q": name, "a": ans, "source": "jsonld"})
    return qas

def _extract_faq_from_visible(page: Dict[str, Any]) -> List[Dict[str, str]]:
    # Prefer explicit v.q/a pairs from crawler if present
    for key in ["visible_faq", "visible_qas", "faq_visible", "visibleFAQ", "visible"]:
        v = page.get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict) and "q" in v[0] and "a" in v[0]:
            out = []
            for item in v:
                q = _escape_ascii(str(item.get("q","")))
                a = _escape_ascii(str(item.get("a","")))
                if q and a:
                    out.append({"q": q, "a": a, "source": "visible"})
            return out

    # Conservative fallback (may over-include; gets cleaned later)
    qas: List[Dict[str, str]] = []
    heads = (page.get("h2") or []) + (page.get("h3") or [])
    pars = page.get("paragraphs") or []
    # Map each heading to the next sufficiently long paragraph
    for q in heads:
        q1 = _escape_ascii(str(q or ""))
        if not q1:
            continue
        a1 = ""
        for p in pars:
            p1 = _escape_ascii(p)
            if len(p1) > 20:
                a1 = p1
                break
        if q1 and a1:
            qas.append({"q": q1, "a": a1, "source": "visible"})
    return qas

def _is_faq_page(url: str,
                 page: Dict[str, Any],
                 schema_qas: List[Dict[str,str]],
                 visible_qas: List[Dict[str,str]],
                 faq_paths: Optional[List[str]]) -> bool:
    path = urlparse(url).path.lower()
    if faq_paths:
        for pat in faq_paths:
            if pat and pat.lower() in path:
                return True
        return False
    if "/faq" in path:
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

# ============= Answer cleaning & rewrite =============

_HEADING_NOISE = re.compile(r"\bfaq\b|\bquick answers\b|\bmanagement summary\b|\bhome\b", re.I)

def _preclean_answer(q: str, a: str) -> str:
    a0 = _escape_ascii(a)
    # drop known heading/noise fragments at start
    a0 = re.sub(r"^\s*(faq\s*[-—:]*\s*)+", "", a0, flags=re.I)
    # drop generic section labels
    if _HEADING_NOISE.search(a0[:80]):
        a0 = _HEADING_NOISE.sub("", a0).strip(" -—:,.")
    # remove question restatement prefix if present
    a0 = _strip_question_restate(q, a0)
    return a0

def _postclean_answer(q: str, a: str, max_words: int) -> str:
    a1 = _escape_ascii(a)
    # if LLM echoed headings or question, strip again
    a1 = re.sub(r"^\s*faq[\s:—-]+.*", "", a1, flags=re.I).strip()
    a1 = _strip_question_restate(q, a1)
    # kill leftovers like “quick answers”
    if _HEADING_NOISE.search(a1[:80]):
        a1 = _HEADING_NOISE.sub("", a1).strip(" -—:,.")
    # enforce one or two sentences max
    parts = re.split(r"(?<=[\.\!\?\:])\s+", a1)
    if parts:
        head = parts[0]
        if len(head.split()) < 6 and len(parts) > 1:
            head = (head + " " + parts[1]).strip()
        a1 = head
    # hard trim
    words = a1.split()
    if len(words) > max_words:
        a1 = " ".join(words[:max_words])
    # if too short after cleaning, fallback to deterministic
    if len(a1.split()) < 3:
        a1 = _deterministic_rewrite(q, a, max_words)
    return a1

def _first_sentence(s: str) -> str:
    s = _escape_ascii(s)
    parts = re.split(r"(?<=[\.\!\?\:])\s+", s)
    return parts[0] if parts else s

def _strip_question_restate(q: str, a: str) -> str:
    qn = _norm(q)
    an = _norm(a)
    if not qn or not an:
        return _escape_ascii(a)
    # if the answer begins with the question or a large prefix of it, strip
    if an.startswith(qn[: max(10, len(qn)//2)]):
        # strip up to first punctuation or the length of q
        idx = len(q)
        return _escape_ascii(a[idx:]).lstrip("-—:,. ").strip()
    return _escape_ascii(a)

def _trim_words(s: str, max_words: int) -> str:
    words = _escape_ascii(s).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])

def _deterministic_rewrite(q: str, a: str, max_words: int) -> str:
    a0 = _preclean_answer(q, a)
    parts = re.split(r"(?<=[\.\!\?\:])\s+", a0)
    if parts:
        head = parts[0]
        if len(head.split()) < 6 and len(parts) > 1:
            head = (head + " " + parts[1]).strip()
        a0 = head
    a0 = _trim_words(a0, max_words)
    return a0

def _llm_rewrite(q: str, a: str, max_words: int) -> Optional[str]:
    try:
        import llm  # project-local interface
    except Exception:
        return None
    pre = _preclean_answer(q, a)
    prompt = (
        "Rewrite the answer below in the SAME language as the text. Rules: ≤{maxw} words, lead with the answer, "
        "do not restate the question, do not include headings like 'FAQ' or 'quick answers', avoid marketing fluff.\n"
        "Question: {q}\n"
        "Answer: {a}\n"
        "Rewrite:"
    ).format(maxw=max_words, q=q, a=pre)
    try:
        model = os.getenv("AEO_LLM_MODEL") or None
        temp = float(os.getenv("AEO_LLM_TEMPERATURE") or 0.1)
        out = None
        if hasattr(llm, "generate"):
            out = llm.generate(prompt, system="You are a precise copy editor.", model=model, temperature=temp, max_tokens=220)
            txt = out.get("text") if isinstance(out, dict) else str(out)
        elif hasattr(llm, "complete"):
            out = llm.complete(prompt, model=model, temperature=temp, max_tokens=220)
            txt = out.get("text") if isinstance(out, dict) else str(out)
        else:
            txt = None
        if txt:
            return _postclean_answer(q, txt, max_words)
    except Exception:
        return None
    return None

def _validate_or_fallback(q: str, original_a: str, candidate_a: str, max_words: int) -> str:
    a = _escape_ascii(candidate_a or "")
    bad = False
    if not a:
        bad = True
    if _HEADING_NOISE.search(a[:80]):
        bad = True
    if _norm(q)[: max(10, len(_norm(q))//2)] in _norm(a)[:80]:
        bad = True
    if len(a.split()) > max_words:
        bad = True
    if bad:
        return _deterministic_rewrite(q, original_a, max_words)
    return a

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
                  dedup_threshold: float,
                  faq_paths: Optional[List[str]]) -> Optional[Dict[str, Any]]:

    schema_qas = _extract_faq_from_jsonld(page)
    visible_qas = _extract_faq_from_visible(page)

    has_schema = bool(schema_qas)
    is_faq = _is_faq_page(url, page, schema_qas, visible_qas, faq_paths)
    page_type = "faq" if is_faq else "other"

    if only_faq and not is_faq:
        # Minimal metrics for non-FAQ pages
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

    # Merge sources (visible baseline; add schema-only)
    merged: List[Dict[str,str]] = []
    for qa in visible_qas:
        merged.append({"q": qa["q"], "a": qa["a"], "src": "visible"})
    vis_norm = set(_norm(qa["q"]) for qa in visible_qas)
    for qa in schema_qas:
        if _norm(qa["q"]) not in vis_norm:
            merged.append({"q": qa["q"], "a": qa["a"], "src": "jsonld"})

    # Dedup clustering
    clusters, canon = _cluster_near_dups(merged, threshold=dedup_threshold)
    canonical_qas: List[Dict[str,str]] = []
    for group, ci in zip(clusters, canon):
        base = merged[ci]
        canonical_qas.append(dict(base))

    # Parity on originals
    parity_ok, miss_in_schema, miss_in_visible = _parity(visible_qas, schema_qas)

    # Improve answers with strict guards
    improved_qas: List[Dict[str,str]] = []
    for qa in canonical_qas:
        q = qa["q"]
        a = qa["a"]
        if emit_improved:
            if rewrite_mode == "llm":
                cand = _llm_rewrite(q, a, max_words)
                cand = _validate_or_fallback(q, a, cand or "", max_words)
            else:
                cand = _deterministic_rewrite(q, a, max_words)
        else:
            cand = _escape_ascii(a)
        improved_qas.append({"q": _escape_ascii(q), "a": cand})

    # Metrics & issues
    n_qas = len(improved_qas)
    overlong = sum(1 for qa in improved_qas if len(_escape_ascii(qa["a"]).split()) > max_words)
    dup_count = sum(max(0, len(g)-1) for g in clusters)
    score, issues = _score_and_issues("faq", has_schema, n_qas, overlong, dup_count, parity_ok)

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
    faq_paths = payload.get("faq_paths") if isinstance(payload.get("faq_paths"), list) else None

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
            dedup_threshold=dedup_threshold,
            faq_paths=faq_paths
        )
        if analyzed:
            out_pages.append(analyzed)

    return {
        "site": {"url": _norm_url(start_url or (pages[0].get("url") if pages else ""))},
        "pages": out_pages
    }
