# aeo_agent.py — AEO analyzer with schema/faq-agent priority, strict cleaning,
# parity checks, near-dup clustering, scoring, and keyword-based suggestions.
#
# Payload options:
#   only_faq: bool = True
#   emit_improved: bool = True
#   rewrite: "deterministic" | "llm" = "deterministic"
#   max_answer_words: int = 80
#   dedup_threshold: float = 0.88
#   faq_paths: [str,...]  # e.g. ["/faq"]
#
# Output (top-level):
# {
#   "site": {"url": "..."},
#   "pages": [{
#     "url": "...", "type": "faq"|"other", "score": int, "issues": [...],
#     "metrics": {
#       "qas_detected": int, "answers_leq_80w": int, "answers_gt_80w": int,
#       "parity_ok": bool, "src_counts": {"jsonld": int, "visible": int, "faq_job": int},
#       "dup_questions": int, "has_faq_schema_detected": bool,
#       "missing_in_schema": [...], "missing_in_visible": [...], "flags": [...]
#     },
#     "qas": [{"q":"...","a":"..."}],
#     "suggested_qas": [{"q":"...","why":"..."}]
#   }]
# }

import html
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from psycopg.rows import dict_row

# ================= Core utils =================

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
    # common mojibake
    t = t.replace("â", "'").replace("â", '"').replace("â", '"')
    t = t.replace("â", "-").replace("â", "-")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _norm(s: str) -> str:
    return _escape_ascii(s).lower()

def _norm_url(u: str) -> str:
    try:
        p = urlparse((u or "").strip())
        scheme = p.scheme or "https"
        host = p.netloc.lower()
        path = p.path or "/"
        q = f"?{p.query}" if p.query else ""
        return f"{scheme}://{host}{path}{q}"
    except Exception:
        return (u or "").strip()

def _has_mojibake(s: str) -> bool:
    return "â" in s or "\ufffd" in s

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

# ================= Extractors =================

def _extract_faq_from_jsonld(page: Dict[str, Any]) -> List[Dict[str, str]]:
    qas: List[Dict[str, str]] = []
    jsonlds = page.get("jsonld") or page.get("ld_json") or []
    if isinstance(jsonlds, dict):
        jsonlds = [jsonlds]
    for j in jsonlds:
        if not isinstance(j, dict): 
            continue
        t = j.get("@type")
        tlist = [t] if isinstance(t, str) else (t or [])
        if any(str(x).lower() == "faqpage" for x in tlist):
            ents = j.get("mainEntity") or []
            if isinstance(ents, dict): 
                ents = [ents]
            for e in ents:
                q = _escape_ascii(str(e.get("name") or e.get("question") or ""))
                aa = e.get("acceptedAnswer") or {}
                if isinstance(aa, list): 
                    aa = aa[0] if aa else {}
                a = _escape_ascii(str(aa.get("text") or aa.get("answer") or ""))
                if q and a:
                    qas.append({"q": q, "a": a, "source": "jsonld"})
    return qas

def _extract_faq_from_visible(page: Dict[str, Any]) -> List[Dict[str, str]]:
    # Prefer explicit visible Q/A if crawler produced that
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
    # Conservative fallback: headings + first decent paragraph
    qas: List[Dict[str, str]] = []
    heads = (page.get("h2") or []) + (page.get("h3") or [])
    pars = page.get("paragraphs") or []
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
        return any(pat and pat.lower() in path for pat in faq_paths)
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

# ================= Dedup & parity =================

_STOP = set("the a an to for and or but with in on at from of by as is are was were be been being this that it its your our their you we they i me my mine".split())

def _tokenize(s: str) -> List[str]:
    s = _norm(s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return [t for t in s.split() if t and t not in _STOP]

def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0

def _cluster_near_dups(qas: List[Dict[str,str]], threshold: float) -> Tuple[List[List[int]], List[int]]:
    n = len(qas); used = [False]*n; clusters=[]; canon=[]
    for i in range(n):
        if used[i]: 
            continue
        base = _tokenize(qas[i]["q"]); group=[i]; used[i]=True
        for j in range(i+1, n):
            if used[j]: 
                continue
            if _jaccard(base, _tokenize(qas[j]["q"])) >= threshold:
                group.append(j); used[j]=True
        clusters.append(group); canon.append(group[0])
    return clusters, canon

def _parity(visible_qas: List[Dict[str,str]], schema_qas: List[Dict[str,str]]) -> Tuple[bool, List[str], List[str]]:
    vis_set = set(_norm(x["q"]) for x in visible_qas)
    sch_set = set(_norm(x["q"]) for x in schema_qas)
    missing_in_schema = [x for x in vis_set if x not in sch_set]
    missing_in_visible = [x for x in sch_set if x not in vis_set]
    return (not missing_in_schema and not missing_in_visible), missing_in_schema, missing_in_visible

# ================= Cleaning & rewrite =================

_HEADING_NOISE = re.compile(r"\bfaq\b|\bquick answers\b|\bour approach\b|\bgenerative seo\b", re.I)

def _strip_question_restate(q: str, a: str) -> str:
    qn = _norm(q); an = _norm(a)
    if not qn or not an: 
        return _escape_ascii(a)
    if an.startswith(qn[: max(10, len(qn)//2)]):
        idx = len(q)
        return _escape_ascii(a[idx:]).lstrip("-—:,. ").strip()
    return _escape_ascii(a)

def _remove_heading_blocks(a: str) -> str:
    s = _escape_ascii(a)
    s = re.sub(r'^\s*(faq[^\.!\?]{0,200}[\.!\?]\s*)', '', s, flags=re.I)
    s = re.sub(r'^\s*(faq\s*[-—:]*\s*)+', '', s, flags=re.I)
    s = re.sub(r'^\s*[^\.!\?]{0,200}\bquick answers\b[:\-\—]?\s*', '', s, flags=re.I)
    s = _HEADING_NOISE.sub("", s).strip(" -—:,.")
    return s

def _preclean_answer(q: str, a: str) -> str:
    a0 = _escape_ascii(a)
    a0 = _remove_heading_blocks(a0)
    a0 = _strip_question_restate(q, a0)
    return a0

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
    return _trim_words(a0, max_words)

def _postclean_answer(q: str, a: str, max_words: int) -> str:
    a1 = _remove_heading_blocks(a)
    a1 = _strip_question_restate(q, a1)
    parts = re.split(r"(?<=[\.\!\?\:])\s+", a1)
    if parts:
        head = parts[0]
        if len(head.split()) < 6 and len(parts) > 1:
            head = (head + " " + parts[1]).strip()
        a1 = head
    a1 = _trim_words(a1, max_words)
    if len(a1.split()) < 3:
        a1 = _deterministic_rewrite(q, a, max_words)
    return a1

def _llm_rewrite(q: str, a: str, max_words: int) -> Optional[str]:
    try:
        import llm
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
    if _HEADING_NOISE.search(a[:120]):
        bad = True
    if _norm(q)[: max(10, len(_norm(q))//2)] in _norm(a)[:120]:
        bad = True
    if len(a.split()) > max_words:
        bad = True
    if bad:
        return _deterministic_rewrite(q, original_a, max_words)
    return a

# ================= FAQ-agent & keywords-agent integration =================

def _harvest_faq_agent_pairs(faq_job_out: Dict[str, Any]) -> List[Dict[str,str]]:
    pairs: List[Dict[str,str]] = []
    if not faq_job_out:
        return pairs
    # Common shapes: {"pairs":[{q,a,source}]}, {"faqs":[... with q,a]}, {"first3":[...]} etc.
    for key in ["pairs", "qas", "faqs", "first3"]:
        arr = faq_job_out.get(key)
        if isinstance(arr, list):
            for it in arr:
                q = _escape_ascii(str(it.get("q","") or it.get("question","")))
                a = _escape_ascii(str(it.get("a","") or it.get("answer","")))
                src = _escape_ascii(str(it.get("source","") or it.get("url","")))
                if q and a:
                    pairs.append({"q": q, "a": a, "source": src})
    # Dedup by normalized question
    seen = set(); out=[]
    for qa in pairs:
        nq = _norm(qa["q"])
        if nq in seen: 
            continue
        seen.add(nq); out.append(qa)
    return out

def _keywords_suggestions(kw_out: Dict[str, Any], existing_qs: List[str], limit: int = 5) -> List[Dict[str,str]]:
    if not kw_out:
        return []
    existing = set(_norm(q) for q in existing_qs)
    suggs = []
    # Attempt to use kw_out["suggestions"] list of {page_title, grouped_keywords, notes}
    arr = kw_out.get("suggestions") or []
    for s in arr:
        gk = s.get("grouped_keywords") or []
        if not gk: 
            continue
        # Turn first grouped keyword into a candidate question if not covered
        head = str(gk[0])
        if _norm(head) in existing:
            continue
        q = f"What about {head}?"
        suggs.append({"q": _escape_ascii(q), "why": _escape_ascii(s.get("notes") or "Keyword cluster gap")})
        if len(suggs) >= limit:
            break
    return suggs

def _best_match_answer_from_repo(q: str, repo: List[Dict[str,str]], threshold: float = 0.88) -> Optional[str]:
    tq = _tokenize(q)
    best = (0.0, None)
    for item in repo:
        sim = _jaccard(tq, _tokenize(item["q"]))
        if sim > best[0]:
            best = (sim, item)
    if best[0] >= threshold and best[1]:
        return best[1]["a"]
    return None

# ================= Scoring =================

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

# ================= Per-page analysis =================

def _analyze_page(page: Dict[str, Any],
                  url: str,
                  only_faq: bool,
                  emit_improved: bool,
                  rewrite_mode: str,
                  max_words: int,
                  dedup_threshold: float,
                  faq_paths: Optional[List[str]],
                  faq_repo: List[Dict[str,str]]) -> Optional[Dict[str, Any]]:

    schema_qas = _extract_faq_from_jsonld(page)
    visible_qas = _extract_faq_from_visible(page)

    has_schema = bool(schema_qas)
    is_faq = _is_faq_page(url, page, schema_qas, visible_qas, faq_paths)
    page_type = "faq" if is_faq else "other"

    # Source priority: schema → faq_repo → visible
    base_candidates: List[Dict[str,str]] = []
    if is_faq and schema_qas:
        base_candidates = [{"q": qa["q"], "a": qa["a"], "src": "jsonld"} for qa in schema_qas]
    elif is_faq and faq_repo:
        # take repo answers that match the page questions if visible provides questions
        if visible_qas:
            for v in visible_qas:
                repo_a = _best_match_answer_from_repo(v["q"], faq_repo, threshold=0.88)
                if repo_a:
                    base_candidates.append({"q": v["q"], "a": repo_a, "src": "faq_repo"})
        if not base_candidates and visible_qas:
            base_candidates = [{"q": qa["q"], "a": qa["a"], "src": "visible"} for qa in visible_qas]
    else:
        # non-FAQ: optionally capture small Q&A if present (soft scoring)
        if not only_faq and visible_qas:
            base_candidates = [{"q": qa["q"], "a": qa["a"], "src": "visible"} for qa in visible_qas]

    if only_faq and not is_faq:
        n = len(base_candidates)
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

    # If still empty on a FAQ page, try visible as last resort
    if is_faq and not base_candidates and visible_qas:
        base_candidates = [{"q": qa["q"], "a": qa["a"], "src": "visible"} for qa in visible_qas]

    # Dedup clusters on questions
    clusters, canon = _cluster_near_dups(base_candidates, threshold=dedup_threshold)
    canonical_qas: List[Dict[str,str]] = []
    for group, ci in zip(clusters, canon):
        canonical_qas.append(dict(base_candidates[ci]))

    # Parity (visible vs schema on the raw extracts)
    parity_ok, miss_in_schema, miss_in_visible = _parity(visible_qas, schema_qas)

    # Improve answers with strict validation
    improved_qas: List[Dict[str,str]] = []
    for qa in canonical_qas:
        q = qa["q"]; a = qa["a"]
        if emit_improved:
            if rewrite_mode == "llm":
                cand = _llm_rewrite(q, a, max_words)
                cand = _validate_or_fallback(q, a, cand or "", max_words)
            else:
                cand = _deterministic_rewrite(q, a, max_words)
        else:
            cand = _escape_ascii(a)
        improved_qas.append({"q": _escape_ascii(q), "a": cand})

    n_qas = len(improved_qas)
    overlong = sum(1 for qa in improved_qas if len(_escape_ascii(qa["a"]).split()) > max_words)
    dup_count = sum(max(0, len(g)-1) for g in clusters)
    score, issues = _score_and_issues(page_type, has_schema, n_qas, overlong, dup_count, parity_ok)

    flags = []
    if any(_has_mojibake(qa["a"]) or _has_mojibake(qa["q"]) for qa in canonical_qas):
        flags.append("mojibake")
    if any("??" in qa["a"] for qa in canonical_qas):
        flags.append("heading_bleed")
    if any(len(_escape_ascii(qa["a"]).split()) < 3 for qa in canonical_qas):
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

    # Keyword-based suggestions (lightweight)
    suggested_qas: List[Dict[str,str]] = []
    # Existing questions to avoid dupe suggestions
    existing_qs = [qa["q"] for qa in improved_qas]

    return {
        "url": url,
        "type": page_type,
        "score": score,
        "issues": issues,
        "metrics": m,
        "qas": improved_qas,
        "suggested_qas": suggested_qas
    }

# ================= Public entrypoint =================

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

    # Pull faq-agent output (if available) to use as authoritative repo
    faq_job_out = _fetch_latest(conn, site_id, "faq") or {}
    faq_repo = _harvest_faq_agent_pairs(faq_job_out)

    # Pull keywords for optional suggestions
    kw_out = _fetch_latest(conn, site_id, "keywords") or {}

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
            faq_paths=faq_paths,
            faq_repo=faq_repo
        )
        if not analyzed:
            continue

        # Attach keyword suggestions per page (light: 0-5 unique gaps)
        existing_qs = [qa["q"] for qa in analyzed.get("qas", [])]
        suggestions = _keywords_suggestions(kw_out, existing_qs, limit=5)
        if suggestions:
            analyzed["suggested_qas"] = suggestions

        out_pages.append(analyzed)

    return {
        "site": {"url": _norm_url(start_url or (pages[0].get("url") if pages else ""))},
        "pages": out_pages
    }
