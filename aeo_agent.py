# aeo_agent.py — Extract & grade site FAQs for AEO (visible + JSON-LD) with rewrite + provenance
# Output shape:
# {
#   "site": {"url": "..."},
#   "pages": [
#     {
#       "url": "...", "type": "faq"|"other", "score": int, "issues": [..],
#       "metrics": {
#         "src_counts": {"jsonld": int, "visible": int, "faq_job": int},
#         "qas_detected": int, "answers_leq_80w": int,
#         "contamination_hits": int, "parity_ok": bool, "has_faq_schema_detected": bool
#       },
#       "qas": [{"q": "...", "a": "...", "source": "visible|jsonld|improved", "status": "OK|Needs rewrite"}]
#     }, ...
#   ]
# }
import os, re, html, json
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from psycopg.rows import dict_row

# ---------- Helpers ----------

CONTAM_PATTERNS = [
    r"(?i)\bfac\b", r"(?i)\bfaq\b", r"(?i)\bquick answers\b",
    r"(?i)\bgenerative\s+seo\s+—?\s+quick answers\b",
]
STOP_TOKENS = re.compile("|".join(CONTAM_PATTERNS))

def _h(s: str) -> str:
    return html.escape(str(s or ""))

def _escape_ascii(s: str) -> str:
    if s is None:
        return ""
    t = html.unescape(str(s))
    repl = {
        "\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00A0": " ",
        "\u200b": ""
    }
    for k,v in repl.items():
        t = t.replace(k, v)
    t = re.sub(r"[ \t]+", " ", t).strip()
    return t

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

def _word_count(s: str) -> int:
    return len(_escape_ascii(s).split())

def _ends_with_punct(s: str) -> bool:
    return bool(re.search(r"[\.!\?]$", _escape_ascii(s)))

def _is_question(s: str) -> bool:
    s2 = _escape_ascii(s)
    return s2.endswith("?") and len(s2) >= 3 and len(s2) <= 180

def _clean_contamination(s: str) -> Tuple[str, bool]:
    t = _escape_ascii(s)
    contaminated = bool(STOP_TOKENS.search(t))
    if contaminated:
        # verwijder bekende kop-/breadcrumb-prefixen die soms in de eerste regels lekken
        t = re.sub(r"(?i)^faq\s*[-–:]\s*", "", t)
        t = re.sub(r"(?i)generative\s*seo\s*[-–:]\s*quick answers\s*", "", t)
        t = re.sub(r"(?i)quick answers\s*[-–:]\s*", "", t)
        t = re.sub(STOP_TOKENS, "", t)
        t = re.sub(r"\s{2,}", " ", t).strip()
    return t, contaminated

def _dedup_qas(qas: List[Dict[str,str]], threshold: float) -> List[Dict[str,str]]:
    seen = []
    out = []
    for qa in qas:
        q = (_escape_ascii(qa.get("q",""))).lower()
        is_dup = any(_jaccard(q, s) >= threshold for s in seen)
        if not is_dup:
            seen.append(q)
            out.append(qa)
    return out

def _jaccard(a: str, b: str) -> float:
    A = set(re.findall(r"\w+", a))
    B = set(re.findall(r"\w+", b))
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def _grade_answer(a: str, max_words: int) -> Tuple[str, List[str]]:
    flags = []
    wc = _word_count(a)
    if wc == 0:
        flags.append("empty")
    if wc > max_words:
        flags.append("over_80w")
    if re.fullmatch(r"(?i)\s*(yes|no)\.?\s*", _escape_ascii(a)):
        flags.append("too_short_yes_no")
    if not _ends_with_punct(a):
        flags.append("no_final_punct")
    status = "OK" if (wc > 0 and wc <= max_words and "too_short_yes_no" not in flags) else "Needs rewrite"
    return status, flags

def _rule_rewrite(q: str, a: str, max_words: int) -> str:
    q2 = _escape_ascii(q)
    a2 = _escape_ascii(a)
    # simpele antwoord-first rewrite:
    # - definities: start met term + “is …”
    if re.match(r"(?i)what\s+is\s+", q2):
        term = re.sub(r"(?i)^what\s+is\s+", "", q2).strip(" ?")
        base = f"{term} is " if term else ""
        body = re.sub(r"(?i)^yes[\.!\s]*", "", a2)
        body = re.sub(r"(?i)^no[\.!\s]*", "", body)
        cand = f"{base}{body}"
    # - yes/no vragen: behoud yes/no + 1 reden
    elif re.match(r"(?i)(do|does|is|are|can|should|will)\b", q2):
        if re.search(r"(?i)\byes\b", a2):
            cand = "Yes. " + re.sub(r"(?i)^yes[\.!\s]*", "", a2)
        elif re.search(r"(?i)\bno\b", a2):
            cand = "No. " + re.sub(r"(?i)^no[\.!\s]*", "", a2)
        else:
            cand = a2
    else:
        cand = a2

    # strip contaminatie nogmaals en cap op max woorden
    cand, _ = _clean_contamination(cand)
    words = cand.split()
    if len(words) > max_words:
        cand = " ".join(words[:max_words]) + "."
    if not _ends_with_punct(cand):
        cand = cand.rstrip() + "."
    return cand

def _try_llm_rewrite(q: str, a: str, max_words: int, model: Optional[str], temperature: float) -> Optional[str]:
    try:
        import llm
    except Exception:
        return None
    prompt = (
        "Rewrite the answer in ≤ {n} words, answer-first, factual, no marketing, end with a period.\n"
        "Q: {q}\nA: {a}\nRewritten:".format(n=max_words, q=q, a=a)
    )
    try:
        if hasattr(llm, "generate"):
            out = llm.generate(prompt, system="You are a concise SEO/AEO editor.", model=model, temperature=temperature, max_tokens=120)
            txt = out.get("text") if isinstance(out, dict) else str(out)
            return _escape_ascii(txt)
    except Exception:
        pass
    try:
        if hasattr(llm, "complete"):
            out = llm.complete(prompt, model=model, temperature=temperature, max_tokens=120)
            txt = out.get("text") if isinstance(out, dict) else str(out)
            return _escape_ascii(txt)
    except Exception:
        pass
    return None

# ---------- Extraction ----------

def _extract_jsonld_faq(page: Dict[str,Any]) -> List[Dict[str,str]]:
    qas = []
    for item in (page.get("jsonld") or []):
        try:
            t = item.get("@type") or item.get("@graph", [{}])[0].get("@type")
        except Exception:
            t = None
        if not t:
            continue
        # flatten if @graph
        candidates = []
        if isinstance(item, dict) and item.get("@type") == "FAQPage":
            candidates = [item]
        if isinstance(item, dict) and "@graph" in item:
            for g in item["@graph"]:
                if isinstance(g, dict) and g.get("@type") == "FAQPage":
                    candidates.append(g)
        for faq in candidates:
            for ent in faq.get("mainEntity") or []:
                if not isinstance(ent, dict): continue
                q = ent.get("name") or ent.get("question") or ""
                ans = ent.get("acceptedAnswer") or {}
                a = ans.get("text") or ""
                if q and a:
                    qas.append({"q": _escape_ascii(q), "a": _escape_ascii(a), "source": "jsonld"})
    return qas

def _extract_visible_faq(page: Dict[str,Any]) -> List[Dict[str,str]]:
    # 1) Als crawler reeds q/a paren heeft geplaatst
    for k in ("faq_visible", "qna_visible", "visible_qna", "faqs"):
        v = page.get(k)
        if isinstance(v, list) and v and isinstance(v[0], dict) and "q" in v[0] and "a" in v[0]:
            return [{"q": _escape_ascii(x["q"]), "a": _escape_ascii(x["a"]), "source": "visible"} for x in v]

    # 2) Heuristische extractie uit H2/H3/paragraphs
    blocks: List[str] = []
    for k in ("h2","h3","paragraphs","p","li"):
        vals = page.get(k) or []
        blocks.extend([_escape_ascii(x) for x in vals if isinstance(x,str) and x.strip()])

    qas: List[Dict[str,str]] = []
    i = 0
    while i < len(blocks):
        if _is_question(blocks[i]):
            q = blocks[i]
            # neem eerstvolgende blok(ken) die geen vraag is als antwoord (1–2 blokken)
            ans_parts = []
            j = i + 1
            while j < len(blocks) and not _is_question(blocks[j]) and len(ans_parts) < 2:
                ans_parts.append(blocks[j])
                j += 1
            a = " ".join(ans_parts).strip()
            if q and a:
                qas.append({"q": q, "a": a, "source": "visible"})
            i = j
        else:
            i += 1
    return qas

# ---------- Page processing ----------

def _process_page(page: Dict[str,Any], opts: Dict[str,Any]) -> Dict[str,Any]:
    url = _norm_url(page.get("url") or "")
    max_words = int(opts.get("max_answer_words", 80))
    dedup_thr = float(opts.get("dedup_threshold", 0.88))
    rewrite_mode = (opts.get("rewrite") or "rule").lower()  # "none"|"rule"|"llm"
    only_faq = bool(opts.get("only_faq"))
    faq_paths = set(opts.get("faq_paths") or [])

    # type detectie
    ptype = "faq" if ("/faq" in url or re.search(r"(?i)\bf(aq|requently-asked)\b", " ".join((page.get("h1") or []) + (page.get("h2") or [])))) else "other"
    if faq_paths:
        ptype = "faq" if any(urlparse(url).path.startswith(p) for p in faq_paths) else ptype
    if only_faq and ptype != "faq":
        return {"url": url, "type": ptype, "score": 15, "issues": ["No Q&A section on page."], "metrics": {"src_counts":{"jsonld":0,"visible":0,"faq_job":0}, "qas_detected":0, "answers_leq_80w":0, "contamination_hits":0, "parity_ok": True, "has_faq_schema_detected": False}, "qas": []}

    visible_qas = _extract_visible_faq(page)
    jsonld_qas = _extract_jsonld_faq(page)
    has_schema = bool(jsonld_qas)

    # Combineer, dedupe (voorkeur: visible > jsonld)
    all_qas = visible_qas + [x for x in jsonld_qas if x["q"] not in {v["q"] for v in visible_qas}]
    all_qas = _dedup_qas(all_qas, dedup_thr)

    contamination_hits = 0
    graded_qas = []
    answers_leq = 0

    # Rewrite / grading
    for qa in all_qas:
        q = qa["q"]
        a_raw = qa["a"]
        a_clean, contaminated = _clean_contamination(a_raw)
        if contaminated: contamination_hits += 1

        status, flags = _grade_answer(a_clean, max_words)

        # Rewrite wanneer nodig
        a_out = a_clean
        source = qa.get("source") or "visible"
        if status == "Needs rewrite":
            if rewrite_mode == "llm":
                model = opts.get("llm_model") or os.getenv("AEO_LLM_MODEL")
                temp = float(opts.get("llm_temperature") or os.getenv("AEO_LLM_TEMPERATURE") or 0.2)
                llm_a = _try_llm_rewrite(q, a_clean, max_words, model, temp)
                if llm_a:
                    a_out = llm_a
                    source = "improved"
                    status, flags = _grade_answer(a_out, max_words)
            elif rewrite_mode == "rule":
                a_out = _rule_rewrite(q, a_clean, max_words)
                source = "improved"
                status, flags = _grade_answer(a_out, max_words)

        if _word_count(a_out) <= max_words: answers_leq += 1

        graded_qas.append({
            "q": q,
            "a": a_out,
            "source": source,
            "status": status,
            "flags": flags
        })

    # Metrics & score
    src_counts = {"jsonld": len(jsonld_qas), "visible": len(visible_qas), "faq_job": 0}
    qas_detected = len(graded_qas)
    parity_ok = True
    if has_schema and visible_qas:
        # check ruwe parity (≥70% van visible vragen ook in JSON-LD qua jaccard)
        vis = [x["q"] for x in visible_qas]
        jld = [x["q"] for x in jsonld_qas]
        hits = 0
        for v in vis:
            if any(_jaccard(v.lower(), j.lower()) >= 0.7 for j in jld):
                hits += 1
        parity_ok = (hits >= max(1, int(0.7 * len(vis))))

    issues = []
    if not qas_detected:
        issues.append("No Q&A section on page.")
    if contamination_hits:
        issues.append("Contamination detected.")
    if has_schema is False and ptype == "faq" and visible_qas:
        issues.append("No FAQPage JSON-LD.")
    # too short generic answers
    too_short = sum(1 for qa in graded_qas if "too_short_yes_no" in qa["flags"])
    if too_short:
        issues.append("Over-short answers (Yes/No).")

    # score: start 100 en minpunten:
    score = 100
    if "No Q&A section on page." in issues: score = 15
    score -= 5 * contamination_hits
    score -= 3 * too_short
    if not parity_ok and has_schema: score -= 5
    score = max(15, min(100, score))

    return {
        "url": url,
        "type": ptype,
        "score": score,
        "issues": issues if issues else ["OK"],
        "metrics": {
            "src_counts": src_counts,
            "qas_detected": qas_detected,
            "answers_leq_80w": answers_leq,
            "contamination_hits": contamination_hits,
            "parity_ok": parity_ok,
            "has_faq_schema_detected": has_schema
        },
        "qas": graded_qas
    }

# ---------- Main ----------

def run(conn, job):
    site_id = job["site_id"]
    payload = job.get("payload") or {}
    # opties
    opts = {
        "only_faq": bool(payload.get("only_faq", False)),
        "faq_paths": payload.get("faq_paths") or [],
        "emit_improved": bool(payload.get("emit_improved", True)),
        "rewrite": (payload.get("rewrite") or os.getenv("AEO_REWRITE", "rule")).lower(),  # "none"|"rule"|"llm"
        "max_answer_words": int(payload.get("max_answer_words") or os.getenv("AEO_MAX_ANSWER_WORDS") or 80),
        "dedup_threshold": float(payload.get("dedup_threshold") or os.getenv("AEO_DEDUP_THRESHOLD") or 0.88),
        "llm_model": payload.get("llm_model") or os.getenv("AEO_LLM_MODEL"),
        "llm_temperature": float(payload.get("llm_temperature") or os.getenv("AEO_LLM_TEMPERATURE") or 0.2),
    }

    crawl = _fetch_latest(conn, site_id, "crawl")
    if not crawl:
        raise ValueError("Crawl job missing; run 'crawl' before 'aeo'.")

    start_url = crawl.get("start_url") or ""
    pages = crawl.get("pages") or []
    out_pages: List[Dict[str,Any]] = []

    for p in pages:
        processed = _process_page(p, opts)
        out_pages.append(processed)

    return {
        "site": {"url": _norm_url(start_url)},
        "pages": out_pages
    }
