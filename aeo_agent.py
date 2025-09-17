import os
import re
import json
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urlsplit, urlunsplit
from html import unescape as html_unescape
from psycopg.rows import dict_row

OPENAI_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

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
    host = p.hostname or ""
    path = (p.path or "/")
    if not path:
        path = "/"
    if not path.startswith("/"):
        path = "/" + path
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return urlunsplit((scheme, host, path, "", ""))

_TAG_RE = re.compile(r"<[^>]+>")
def _strip_html(s: str) -> str:
    if not s:
        return ""
    s = html_unescape(s)
    s = _TAG_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _json_obj(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if isinstance(x, str) and x.strip():
        try:
            v = json.loads(x)
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}
    return {}

def _json_arr(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    if isinstance(x, str) and x.strip():
        try:
            v = json.loads(x)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return []

def _detect_lang(texts: List[str], site_lang: Optional[str]) -> str:
    site = (site_lang or "").lower()
    default = "nl" if site.startswith("nl") else "en"
    sample = (" ".join([t for t in texts if t])[:800] + " ").lower()
    nl_score = sum(1 for w in [" de "," het "," een "," en "," voor "," met "," jouw "," je "," wij "," onze "] if w in sample)
    en_score = sum(1 for w in [" the "," and "," for "," with "," your "," we "," our "," to "," of "] if w in sample)
    if nl_score > en_score:
        return "nl"
    if en_score > nl_score:
        return "en"
    return default

QUESTION_PREFIXES = (
    "what ","how ","why ","when ","can ","does ","do ","is ","are ","should ","will ","where ","who ",
    "wat ","hoe ","waarom ","wanneer ","kan ","doet ","doen ","is ","zijn ","moet ","zal ","waar ","wie "
)

def _looks_like_question(s: str) -> bool:
    t = (s or "").strip().lower()
    if not t:
        return False
    if t.endswith("?"):
        return True
    return any(t.startswith(p) for p in QUESTION_PREFIXES) and len(t.split()) >= 2

def _normalize_question(q: str) -> str:
    t = (q or "").strip()
    t = re.sub(r"\s+", " ", t)
    if t and not t.endswith("?"):
        if any(t.lower().startswith(p) for p in QUESTION_PREFIXES):
            t = t.rstrip(".! ") + "?"
    return t

def _trim_words(s: str, limit: int = 80) -> Tuple[str, bool]:
    words = (s or "").split()
    if len(words) <= limit:
        return " ".join(words), False
    return " ".join(words[:limit]) + "…", True

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
        q = _strip_html(q)
        a_text = _strip_html(a_text)
        if q and a_text:
            out.append({"q": _normalize_question(q), "a": a_text})

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
    blocks: List[str] = []
    for key in ("h2","h3","dt","summary","buttons","paragraphs","li"):
        arr = page.get(key) or []
        if isinstance(arr, list):
            blocks.extend([_strip_html(x) for x in arr if isinstance(x, str)])
    blocks = [b for b in blocks if b]
    used = set()
    qas: List[Dict[str,str]] = []
    for i, blk in enumerate(blocks):
        if not _looks_like_question(blk):
            continue
        a = ""
        for j in range(i+1, min(i+6, len(blocks))):
            if j in used:
                continue
            cand = blocks[j]
            if _looks_like_question(cand):
                break
            if len(cand.split()) < 6:
                continue
            a = cand
            used.add(j)
            break
        if a:
            qas.append({"q": _normalize_question(blk), "a": a})
    seen = set()
    out = []
    for qa in qas:
        k = (qa["q"].strip().lower(), qa["a"][:80].strip().lower())
        if k in seen:
            continue
        seen.add(k)
        out.append(qa)
    return out

def _index_faq_job(faq_job_raw: Any) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, bool]]:
    by_url: Dict[str, List[Dict[str, str]]] = {}
    schema_flags: Dict[str, bool] = {}
    job = _json_obj(faq_job_raw)
    arr = job.get("faqs")
    if isinstance(arr, str):
        arr = _json_arr(arr)
    if isinstance(arr, list):
        for it in arr:
            if not isinstance(it, dict):
                continue
            u = _norm_url(it.get("url") or it.get("page") or it.get("source") or "")
            if not u:
                continue
            q = it.get("q") or it.get("question")
            a = it.get("a") or it.get("answer")
            if q and a:
                by_url.setdefault(u, []).append({"q": _normalize_question(q), "a": _strip_html(a)})
            if isinstance(it.get("has_faq_schema"), bool):
                schema_flags[u] = schema_flags.get(u, False) or bool(it["has_faq_schema"])
            qas_list = it.get("qas")
            if isinstance(qas_list, list):
                for qa in qas_list:
                    if not isinstance(qa, dict):
                        continue
                    q2 = qa.get("q") or qa.get("question")
                    a2 = qa.get("a") or qa.get("answer")
                    if q2 and a2:
                        by_url.setdefault(u, []).append({"q": _normalize_question(q2), "a": _strip_html(a2)})
    return by_url, schema_flags

def _is_likely_faq_url(u: str) -> bool:
    p = urlsplit(u)
    path = (p.path or "").lower()
    frag = (p.fragment or "").lower()
    joined = path + ("#" + frag if frag else "")
    for hint in ("/faq", "/faqs", "/veelgestelde-vragen", "#faq"):
        if hint in joined:
            return True
    return False

def _infer_ptype(url: str, page: Dict[str, Any]) -> str:
    given = (page.get("type") or "").lower().strip()
    if given == "faq":
        return "faq"
    if page.get("faq_jsonld"):
        return "faq"
    metrics = page.get("metrics") or {}
    if bool(metrics.get("has_faq_schema")):
        return "faq"
    if _is_likely_faq_url(url):
        return "faq"
    return "other"

def _score_and_issues(ptype: str, counts: Dict[str, Any], has_faq_schema: bool, dup_q: int, dup_a: int, parity_ok: bool) -> Tuple[int, List[str]]:
    issues: List[str] = []
    score = 100
    qas = int(counts.get("merged", 0))
    gt80 = int(counts.get("answers_gt_80w", 0))
    if qas == 0:
        issues.append("No Q&A section on page.")
        score -= 85
    if ptype == "faq" and qas > 0 and qas < 3:
        issues.append("Too few Q&A (min 3).")
        score -= 20
    if qas > 0 and not has_faq_schema:
        issues.append("No FAQPage JSON-LD.")
        score -= 15
    if gt80 > 0:
        issues.append("Overlong answers (>80 words).")
        score -= min(20, gt80 * 5)
    if dup_q > 0:
        issues.append("Duplicate questions across sources.")
        score -= min(15, dup_q * 3)
    if dup_a > 0:
        issues.append("Duplicate answers across sources.")
        score -= min(10, dup_a * 2)
    if qas > 0 and not parity_ok:
        issues.append("Schema/text parity mismatch.")
        score -= 10
    score = max(0, min(100, score))
    return score, issues

def _merge_qas(q_jsonld: List[Dict[str, str]], q_faqjob: List[Dict[str, str]], q_visible: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, int], int, int]:
    src_tagged: List[Tuple[str, Dict[str,str]]] = []
    for qa in q_jsonld:
        src_tagged.append(("jsonld", qa))
    for qa in q_faqjob:
        src_tagged.append(("faq_job", qa))
    for qa in q_visible:
        src_tagged.append(("visible", qa))
    src_counts = {"jsonld": len(q_jsonld), "faq_job": len(q_faqjob), "visible": len(q_visible)}
    seen_q = {}
    seen_a = {}
    dup_q = 0
    dup_a = 0
    merged: List[Dict[str, str]] = []
    for src, qa in src_tagged:
        q = (qa.get("q") or "").strip()
        a = _strip_html(qa.get("a") or "")
        if not q or not a:
            continue
        qk = q.lower()
        ak = a[:160].strip().lower()
        if qk in seen_q:
            dup_q += 1
            continue
        if ak in seen_a:
            dup_a += 1
            continue
        seen_q[qk] = True
        seen_a[ak] = True
        merged.append({"q": q, "a": a})
    return merged, src_counts, dup_q, dup_a

def _answers_count(qa_list: List[Dict[str, str]]) -> Tuple[int, int]:
    gt = 0
    le = 0
    for qa in qa_list:
        w = len((qa.get("a") or "").split())
        if w > 80:
            gt += 1
        else:
            le += 1
    return gt, le

def _limit_qas(qas: List[Dict[str, str]], limit: Optional[int]) -> List[Dict[str, str]]:
    if not limit or limit <= 0:
        return qas
    return qas[: int(limit)]

def generate_aeo(conn, job):
    site_id = job["site_id"]
    payload = job.get("payload") or {}
    max_pages = int(payload.get("max_pages") or 30)
    qas_per_page = int(payload.get("qas_per_page") or 20)
    site_meta = _fetch_site_meta(conn, site_id)
    crawl = _fetch_latest_job(conn, site_id, "crawl")
    faq_job = _fetch_latest_job(conn, site_id, "faq")
    if not crawl:
        return {"site": site_meta, "pages": []}

    pages_in = crawl.get("pages") or []
    pages: List[Dict[str, Any]] = []

    faq_by_url, faq_schema_flags = _index_faq_job(faq_job)

    for p in pages_in:
        url = _norm_url(p.get("final_url") or p.get("url") or "")
        if not url:
            continue
        pages.append({**p, "url": url})

    seen_urls = set()
    unique_pages: List[Dict[str, Any]] = []
    for p in pages:
        u = p["url"]
        if u in seen_urls:
            continue
        seen_urls.add(u)
        unique_pages.append(p)

    unique_pages = unique_pages[:max_pages]

    out_pages: List[Dict[str, Any]] = []
    for p in unique_pages:
        url = p["url"]
        lang = p.get("lang") or _detect_lang([(p.get("title") or ""), (p.get("h1") or "")] + (p.get("paragraphs") or []), site_meta.get("language"))
        ptype = _infer_ptype(url, p)

        faq_jsonld_raw = p.get("faq_jsonld")
        if isinstance(faq_jsonld_raw, str):
            try:
                faq_jsonld = json.loads(faq_jsonld_raw)
            except Exception:
                faq_jsonld = {}
        else:
            faq_jsonld = faq_jsonld_raw or {}

        q_jsonld = _qas_from_jsonld(faq_jsonld)
        q_faqjob = faq_by_url.get(url, [])
        q_visible = _qas_from_visible(p)

        merged, src_counts, dup_q, dup_a = _merge_qas(q_jsonld, q_faqjob, q_visible)
        gt80, le80 = _answers_count(merged)
        has_faq_schema = False
        types = [str(t).lower() for t in (p.get("jsonld_types") or [])]
        if "faqpage" in types:
            has_faq_schema = True
        if faq_jsonld:
            has_faq_schema = True
        if url in faq_schema_flags and faq_schema_flags[url]:
            has_faq_schema = True

        parity_ok = True
        if (src_counts["visible"] > 0 and src_counts["jsonld"] == 0) or (src_counts["jsonld"] > 0 and src_counts["visible"] == 0):
            parity_ok = False

        counts = {
            "merged": len(merged),
            "answers_gt_80w": gt80,
            "answers_leq_80w": le80
        }
        score, issues = _score_and_issues(ptype, counts, has_faq_schema, dup_q, dup_a, parity_ok)

        limited_qas = _limit_qas(merged, qas_per_page)
        qas_trimmed: List[Dict[str, str]] = []
        for qa in limited_qas:
            a_trim, _ = _trim_words(qa["a"], 80)
            qas_trimmed.append({"q": qa["q"], "a": a_trim})

        metrics = {
            "qas_detected": len(merged),
            "answers_gt_80w": gt80,
            "answers_leq_80w": le80,
            "has_faq_schema_detected": bool(has_faq_schema),
            "src_counts": src_counts,
            "dup_questions": dup_q,
            "dup_answers": dup_a,
            "parity_ok": parity_ok
        }

        out_pages.append({
            "url": url,
            "lang": lang,
            "type": ptype,
            "score": score,
            "issues": issues,
            "metrics": metrics,
            "qas": qas_trimmed,
            "faq_jsonld": faq_jsonld if faq_jsonld else None
        })

    return {
        "site": {
            "url": site_meta.get("url"),
            "language": site_meta.get("language"),
            "country": site_meta.get("country"),
            "account_name": site_meta.get("account_name"),
        },
        "pages": out_pages
    }
