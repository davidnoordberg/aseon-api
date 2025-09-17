# aeo_agent.py — AEO extraction + QA quality scoring + optional LLM rewrites
# Output schema:
# {
#   "site": {"url": "..."},
#   "summary": {...},
#   "pages": [
#     {
#       "url": "...",
#       "type": "faq"|"other",
#       "score": 0-100,
#       "issues": ["..."],
#       "metrics": {
#         "src_counts": {"visible": n, "jsonld": n, "faq_job": n},
#         "qas_detected": n,
#         "answers_leq_80w": n,
#         "parity_ok": True|False,
#         "has_faq_schema_detected": True|False,
#         "duplicates": n,
#         "contamination_hits": n,
#         "needs_rewrite": n
#       },
#       "qas": [
#         {
#           "q": "...",
#           "a": "...",              # originele gekozen answer
#           "source": "visible|jsonld|faq_job",
#           "status": "ok|needs_rewrite",
#           "flags": ["over_80w","contamination","missing_punct","duplicate"],
#           "improved": "...",       # alleen als rewrite wezenlijk verschilt
#           "emitted": "original|improved"
#         }
#       ]
#     }
#   ]
# }

import html
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from difflib import SequenceMatcher
from psycopg.rows import dict_row

# ===== util =====

def _escape_ascii(s: str) -> str:
    if s is None:
        return ""
    # decode HTML entities + normalize punctuation / whitespace
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

def _word_count(s: str) -> int:
    return len((_escape_ascii(s) or "").split())

def _ends_with_punct(s: str) -> bool:
    return bool(re.search(r"[.!?]$", _escape_ascii(s)))

def _strip_contamination(s: str) -> Tuple[str, bool]:
    """Verwijder kop/vervuiling die soms in antwoorden lekt (FAQ, headings, 'quick answers', etc.)."""
    before = _escape_ascii(s)
    s2 = re.sub(r"(?i)\bfaq\b[^.:;]*[:\-–—]?\s*", "", before)
    s2 = re.sub(r"(?i)\bquick answers\b[:\-–—]?\s*", "", s2)
    s2 = re.sub(r"(?i)\bgenerative seo\b\s*[-—–]?\s*", lambda m: "Generative SEO " if m.group(0).islower() else "Generative SEO ", s2)
    s2 = re.sub(r"^\s*[-–—]\s*", "", s2)
    s2 = s2.strip()
    return (s2, s2 != before)

def _trim_to_words(s: str, n: int) -> str:
    words = _escape_ascii(s).split()
    if len(words) <= n:
        return " ".join(words)
    return " ".join(words[:n]) + "…"

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, _escape_ascii(a).lower(), _escape_ascii(b).lower()).ratio()

def _dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        k = _escape_ascii(x).lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out

# ===== DB helper =====

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

# ===== source readers (be tolerant voor verschillende crawl/faq vormen) =====

def _read_visible_qas_from_page(p: Dict[str, Any]) -> List[Dict[str, str]]:
    # bekende sleutels die we eerder gezien hebben
    for key in ["visible_qas", "visible", "sample_visible"]:
        arr = p.get(key)
        if isinstance(arr, list) and arr and isinstance(arr[0], dict) and "q" in arr[0] and "a" in arr[0]:
            return [{"q": _escape_ascii(x.get("q","")), "a": _escape_ascii(x.get("a",""))} for x in arr]
    # fallback: niets
    return []

def _read_jsonld_qas_from_page(p: Dict[str, Any]) -> List[Dict[str, str]]:
    # sommige crawlers stoppen de FAQ JSON-LD uitgesplitst in p["faq_jsonld"] of p["jsonld_faq"]
    for key in ["faq_jsonld", "jsonld_faq", "faq_ld"]:
        arr = p.get(key)
        if isinstance(arr, list) and arr and isinstance(arr[0], dict):
            out = []
            for item in arr:
                q = item.get("q") or item.get("name") or ""
                a = item.get("a") or (item.get("acceptedAnswer") or {}).get("text") or ""
                if q and a:
                    out.append({"q": _escape_ascii(q), "a": _escape_ascii(a)})
            if out:
                return out
    # soms: ruwe JSON-LD blob met FAQPage
    for key in ["jsonld", "ld", "structured_data"]:
        arr = p.get(key)
        if isinstance(arr, list):
            out = []
            for obj in arr:
                try:
                    t = obj.get("@type")
                    if (t == "FAQPage" or (isinstance(t, list) and "FAQPage" in t)) and "mainEntity" in obj:
                        for qn in obj["mainEntity"]:
                            q = qn.get("name") or ""
                            a = ((qn.get("acceptedAnswer") or {}).get("text")) or ""
                            if q and a:
                                out.append({"q": _escape_ascii(q), "a": _escape_ascii(a)})
                except Exception:
                    pass
            if out:
                return out
    return []

def _read_faq_job_map(faq_job: Optional[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    """Maak een mapping url -> [ {q,a}, ... ] uit de faq-agent output, tolerant voor varianten."""
    m: Dict[str, List[Dict[str, str]]] = {}
    if not faq_job:
        return m
    # varianten: faq_job["items"], faq_job["faqs"], etc.
    candidates = []
    if isinstance(faq_job.get("items"), list):
        candidates = faq_job["items"]
    elif isinstance(faq_job.get("faqs"), list):
        candidates = faq_job["faqs"]
    elif isinstance(faq_job.get("data"), list):
        candidates = faq_job["data"]
    # items kunnen {q,a,source:url} bevatten
    for it in candidates:
        url = it.get("source") or it.get("url") or ""
        q = it.get("q") or it.get("name") or ""
        a = it.get("a") or it.get("text") or ""
        if url and q and a:
            url = _norm_url(url)
            m.setdefault(url, []).append({"q": _escape_ascii(q), "a": _escape_ascii(a)})
    return m

# ===== QA evaluation + rewrite =====

CONTAM_RE = re.compile(r"(?i)\bfaq\b|quick answers|generative seo\s*—|\bgseo\b", re.I)

def _eval_and_clean_qa(q: str, a: str, max_words: int) -> Tuple[str, List[str]]:
    flags: List[str] = []
    # strip contamination
    a2, changed = _strip_contamination(a)
    if changed or CONTAM_RE.search(a):
        flags.append("contamination")
    # trim length
    if _word_count(a2) > max_words:
        flags.append("over_80w")
        a2 = _trim_to_words(a2, max_words)
    # punct
    if not _ends_with_punct(a2):
        flags.append("missing_punct")
        a2 = a2.rstrip(".!?") + "."
    return a2, flags

def _needs_rewrite(flags: List[str]) -> bool:
    # alles wat niet puur cosmetisch is → rewrite
    return any(f in {"contamination", "over_80w"} for f in flags)

def _llm_rewrite(q: str, a: str, max_words: int, language: str = "en") -> Optional[str]:
    """Vraag het lokale llm-helper te herschrijven, met strikte guard-rails."""
    try:
        import llm  # project helper
    except Exception:
        return None
    sys = ("You are a careful editor. Rewrite the answer so it is factual, concise, ≤{max_words} words, "
           "answer-first, and free of headings like 'FAQ' or 'quick answers'. Keep the meaning identical and do not add new facts. "
           "End with a period. Language: {lang}.").format(max_words=max_words, lang=language)
    prompt = (
        "Question:\n"
        f"{q}\n\n"
        "Existing answer:\n"
        f"{a}\n\n"
        "Rewrite now."
    )
    try:
        if hasattr(llm, "generate"):
            out = llm.generate(prompt, system=sys, model=os.getenv("AEO_LLM_MODEL") or None, temperature=float(os.getenv("AEO_LLM_T", "0.2")), max_tokens=int(os.getenv("AEO_LLM_MAXTOK", "160")))
            txt = out.get("text") if isinstance(out, dict) else str(out)
        elif hasattr(llm, "chat"):
            msgs = [{"role": "system", "content": sys}, {"role": "user", "content": prompt}]
            out = llm.chat(messages=msgs, model=os.getenv("AEO_LLM_MODEL") or None, temperature=float(os.getenv("AEO_LLM_T", "0.2")), max_tokens=int(os.getenv("AEO_LLM_MAXTOK", "160")))
            txt = out.get("content") if isinstance(out, dict) else str(out)
        else:
            txt = None
    except Exception:
        txt = None
    if not txt:
        return None
    # safety post-process
    txt = _escape_ascii(txt)
    txt, _ = _strip_contamination(txt)
    if _word_count(txt) > max_words:
        txt = _trim_to_words(txt, max_words)
    if not _ends_with_punct(txt):
        txt = txt.rstrip(".!?") + "."
    return txt

# ===== Main AEO generator =====

def generate_aeo(conn, job):
    site_id = job["site_id"]
    payload = job.get("payload") or {}
    only_faq = bool(payload.get("only_faq"))
    faq_paths = payload.get("faq_paths") or []  # e.g., ["/faq"]
    emit_improved = bool(payload.get("emit_improved"))
    rewrite_mode = payload.get("rewrite") or "none"  # "none" | "llm"
    max_answer_words = int(payload.get("max_answer_words") or 80)
    dedup_threshold = float(payload.get("dedup_threshold") or 0.90)
    language = (payload.get("language") or "en").lower()

    crawl = _fetch_latest(conn, site_id, "crawl")
    if not crawl:
        raise ValueError("Crawl job missing; run 'crawl' before 'aeo'.")
    faq_job = _fetch_latest(conn, site_id, "faq")
    faq_map = _read_faq_job_map(faq_job)

    start_url = crawl.get("start_url") or ""
    pages_in = crawl.get("pages") or []
    out_pages: List[Dict[str, Any]] = []

    for p in pages_in:
        url = _norm_url(p.get("url") or "")
        if not url:
            continue
        if only_faq:
            if faq_paths and not any(urlparse(url).path.startswith(fp) for fp in faq_paths):
                continue

        # collect sources
        vis = _read_visible_qas_from_page(p)
        jld = _read_jsonld_qas_from_page(p)
        via_faq_job = faq_map.get(url, [])

        src_counts = {
            "visible": len(vis),
            "jsonld": len(jld),
            "faq_job": len(via_faq_job)
        }

        qas_raw: List[Tuple[str, str, str]] = []  # (q,a,source)
        for qa in vis:
            if qa.get("q") and qa.get("a"):
                qas_raw.append((qa["q"], qa["a"], "visible"))
        for qa in jld:
            if qa.get("q") and qa.get("a"):
                qas_raw.append((qa["q"], qa["a"], "jsonld"))
        for qa in via_faq_job:
            if qa.get("q") and qa.get("a"):
                qas_raw.append((qa["q"], qa["a"], "faq_job"))

        # dedup op vraag, met tolerant threshold
        deduped: List[Tuple[str, str, str]] = []
        seen_q: List[str] = []
        for q, a, s in qas_raw:
            qn = _escape_ascii(q).lower()
            if any(_similar(qn, prev) >= dedup_threshold for prev in seen_q):
                # markeer duplicate via flag later
                deduped.append((q, a, s))  # we houden hem, flaggen we later
            else:
                seen_q.append(qn)
                deduped.append((q, a, s))

        # bepaal pagetype/faq aanwezig
        is_faq = (src_counts["visible"] + src_counts["jsonld"]) >= 3 or urlparse(url).path.rstrip("/").endswith("faq")

        # per QA: schoonmaken, flags, (optioneel) LLM-rewrite
        qas_out: List[Dict[str, Any]] = []
        duplicates = 0
        contamination_hits = 0
        needs_rw = 0
        answers_leq_80w = 0

        # duplicate detect op basis van vraagnorm
        seen_q_norm = {}
        for q, a, s in deduped:
            q0 = _escape_ascii(q)
            a0 = _escape_ascii(a)
            q_norm = q0.lower()
            dup_flag = False
            if q_norm in seen_q_norm and _similar(q_norm, seen_q_norm[q_norm]) >= dedup_threshold:
                dup_flag = True
                duplicates += 1
            else:
                seen_q_norm[q_norm] = q_norm

            cleaned_a, flags = _eval_and_clean_qa(q0, a0, max_answer_words)
            if "contamination" in flags:
                contamination_hits += 1
            if _word_count(cleaned_a) <= max_answer_words:
                answers_leq_80w += 1

            status = "ok"
            if dup_flag:
                flags.append("duplicate")
                status = "needs_rewrite"
            if _needs_rewrite(flags):
                status = "needs_rewrite"
            if status == "needs_rewrite":
                needs_rw += 1

            improved = None
            emitted = "original"
            if emit_improved and (status == "needs_rewrite" or rewrite_mode == "llm"):
                cand = None
                if rewrite_mode == "llm":
                    cand = _llm_rewrite(q0, cleaned_a, max_answer_words, language=language)
                # deterministic fallback als llm niet beschikbaar
                if not cand:
                    cand = cleaned_a
                # alleen tonen als inhoudelijk verschillend
                if _similar(cand, cleaned_a) < 0.97:
                    improved = cand
                    emitted = "improved"
                else:
                    improved = None
                    emitted = "original"

            qas_out.append({
                "q": q0,
                "a": cleaned_a,
                "source": s,
                "status": status,
                "flags": _dedup_keep_order(flags),
                "improved": improved,
                "emitted": emitted
            })

        # page issues + score
        issues: List[str] = []
        if is_faq and src_counts["jsonld"] == 0:
            issues.append("No FAQPage JSON-LD.")
        if duplicates > 0:
            issues.append("Duplicate questions.")
        if contamination_hits > 0:
            issues.append("Heading contamination in answers.")
        if is_faq and (src_counts["visible"] + src_counts["jsonld"]) < 3:
            issues.append("Too few Q&A (min 3).")

        # parity: if jsonld present, check overlap with visible
        def _qset(arr): return set([_escape_ascii(x.get("q","")).lower() for x in arr])
        parity_ok = True
        if src_counts["jsonld"] and src_counts["visible"]:
            inter = len(_qset(jld) & _qset(vis))
            parity_ok = inter >= max(1, int(0.6 * min(len(jld), len(vis))))

        has_faq_schema = bool(src_counts["jsonld"])
        qas_detected = len(qas_out)

        score = 100
        if not is_faq and qas_detected == 0:
            score = 15
        else:
            score -= min(20, duplicates * 5)
            score -= 10 if not has_faq_schema and is_faq else 0
            score -= min(15, contamination_hits * 5)
            # lichte straf als veel rewrites nodig
            score -= min(20, needs_rw * 2)

        out_pages.append({
            "url": url,
            "type": "faq" if is_faq else "other",
            "score": max(15, min(100, score)),
            "issues": issues if issues else ["OK"],
            "metrics": {
                "src_counts": src_counts,
                "qas_detected": qas_detected,
                "answers_leq_80w": answers_leq_80w,
                "parity_ok": parity_ok,
                "has_faq_schema_detected": has_faq_schema,
                "duplicates": duplicates,
                "contamination_hits": contamination_hits,
                "needs_rewrite": needs_rw
            },
            "qas": qas_out
        })

    # filter alleen faq pages indien only_faq
    pages_final = out_pages
    if only_faq:
        pages_final = [p for p in out_pages if p["type"] == "faq"]

    # summary
    site_url = _norm_url(start_url)
    faq_pages = [p for p in pages_final if p["type"] == "faq"]
    faq_count = len(faq_pages)
    total_qas = sum(p["metrics"]["qas_detected"] for p in pages_final)
    total_contam = sum(p["metrics"]["contamination_hits"] for p in pages_final)
    total_needs_rw = sum(p["metrics"]["needs_rewrite"] for p in pages_final)

    return {
        "site": {"url": site_url},
        "summary": {
            "pages": len(pages_final),
            "faq_pages": faq_count,
            "total_qas": total_qas,
            "contamination_hits": total_contam,
            "needs_rewrite": total_needs_rw
        },
        "pages": pages_final
    }
