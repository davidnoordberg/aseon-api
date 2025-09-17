#aeo agent

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
    path = (p.path or "/") or "/"
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
    if t:
        t = t[0].upper() + t[1:]
    return t

def _trim_words(s: str, limit: int = 80) -> Tuple[str, bool]:
    words = (s or "").split()
    if len(words) <= limit:
        return " ".join(words), False
    return " ".join(words[:limit]) + "…", True

def _first_sentence(s: str) -> str:
    s2 = (s or "").strip()
    m = re.search(r'(.+?[.!?])(\s|$)', s2)
    return m.group(1).strip() if m else s2

def _has_marketing_filler(s: str) -> bool:
    t = (s or "").lower()
    bad = ["contact us","get in touch","learn more","click here","our team","we offer","we provide","ask us"]
    return any(b in t for b in bad)

def _clean_answer(a: str) -> str:
    a0 = _strip_html(a)
    a0 = re.sub(r'\s+', ' ', a0).strip()
    return a0

# -------- JSON-LD -> Q/A --------
def _qas_from_jsonld(faq_jsonld_any: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []

    def _handle_entity(entity):
        if not isinstance(entity, dict):
            return
        q = entity.get("name") or entity.get("question") or entity.get("headline") or ""
        ans = entity.get("acceptedAnswer") or entity.get("accepted_answer") or {}
        if isinstance(ans, list) and ans:
            ans = ans[0]
        a_text = ""
        if isinstance(ans, dict):
            a_text = ans.get("text") or ans.get("answer") or ans.get("articleBody") or ""
        q = _strip_html(q)
        a_text = _clean_answer(a_text)
        if q and a_text:
            out.append({"q": _normalize_question(q), "a": a_text, "source": "jsonld"})

    def _handle_root(obj):
        if not isinstance(obj, dict):
            return
        if obj.get("@type") == "FAQPage" or str(obj.get("@type","")).lower() == "faqpage":
            main = obj.get("mainEntity") or obj.get("main_entity") or []
            if isinstance(main, list):
                for ent in main:
                    _handle_entity(ent)
            else:
                _handle_entity(main)
        if "@graph" in obj and isinstance(obj["@graph"], list):
            for node in obj["@graph"]:
                if isinstance(node, dict):
                    _handle_root(node)

    if isinstance(faq_jsonld_any, list):
        for it in faq_jsonld_any:
            if isinstance(it, (dict, list)):
                _handle_root(it) if isinstance(it, dict) else [_handle_root(x) for x in it if isinstance(x, dict)]
    elif isinstance(faq_jsonld_any, dict):
        _handle_root(faq_jsonld_any)

    seen = set()
    deduped: List[Dict[str, str]] = []
    for qa in out:
        key = qa["q"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(qa)
    return deduped

# -------- Zichtbaar -> Q/A (DOM-volgorde eerst) --------
def _extract_visible_from_page(p: Dict[str, Any]) -> List[Dict[str, str]]:
    # 0) Als de crawler al Q/A-paren levert (faq_visible), gebruik die eerst.
    if isinstance(p.get("faq_visible"), list) and p["faq_visible"]:
        out = []
        for item in p["faq_visible"]:
            if not isinstance(item, dict):
                continue
            q = _strip_html(item.get("q") or "")
            a = _clean_answer(item.get("a") or "")
            if q and a:
                out.append({"q": _normalize_question(q), "a": a, "source": "visible"})
        if out:
            return out

    # 1) DOM-volgorde gebruiken als er dom_blocks is
    dom = p.get("dom_blocks")
    if isinstance(dom, list) and dom:
        blocks = []
        for b in dom:
            # zowel {tag,text} als plain tekst ondersteunen
            if isinstance(b, dict):
                txt = _strip_html(b.get("text") or "")
            else:
                txt = _strip_html(str(b))
            if txt:
                blocks.append(txt)
    else:
        # 2) Fallback: verzamel losse lijsten (geen volgorde-garantie)
        blocks = []
        for key in ("h2","h3","dt","dd","summary","buttons","paragraphs","li"):
            arr = p.get(key) or []
            if isinstance(arr, list):
                blocks.extend([_strip_html(x) for x in arr if isinstance(x, str)])
        blocks = [b for b in blocks if b]

    used = set()
    qas: List[Dict[str,str]] = []
    for i, blk in enumerate(blocks):
        if not _looks_like_question(blk):
            continue
        a = ""
        for j in range(i+1, min(i+10, len(blocks))):
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
            qas.append({"q": _normalize_question(blk), "a": _clean_answer(a), "source": "visible"})
    seen = set()
    out = []
    for qa in qas:
        k = (qa["q"].strip().lower(), qa["a"][:120].strip().lower())
        if k in seen:
            continue
        seen.add(k)
        out.append(qa)
    return out

# -------- FAQ job aggregatie --------
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
                by_url.setdefault(u, []).append({"q": _normalize_question(_strip_html(q)), "a": _clean_answer(a), "source": "faq_job"})
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
                        by_url.setdefault(u, []).append({"q": _normalize_question(_strip_html(q2)), "a": _clean_answer(a2), "source": "faq_job"})
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
    if page.get("faq_jsonld") or page.get("faq_visible"):
        return "faq"
    metrics = page.get("metrics") or {}
    if bool(metrics.get("has_faq_schema")):
        return "faq"
    if _is_likely_faq_url(url):
        return "faq"
    return "other"

# -------- Q/A merge + audit --------
def _merge_qas(q_jsonld: List[Dict[str, str]], q_faqjob: List[Dict[str, str]], q_visible: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, int], int, int, List[Dict[str,str]]]:
    all_src = q_jsonld + q_visible + q_faqjob
    src_counts = {"jsonld": len(q_jsonld), "faq_job": len(q_faqjob), "visible": len(q_visible)}
    seen_q = {}
    seen_a = {}
    dup_q = 0
    dup_a = 0
    merged: List[Dict[str, str]] = []
    audit_stream: List[Dict[str,str]] = []
    pref_rank = {"jsonld": 3, "visible": 2, "faq_job": 1}
    by_q: Dict[str, Dict[str,str]] = {}
    for qa in all_src:
        q = (qa.get("q") or "").strip()
        a = _clean_answer(qa.get("a") or "")
        src = qa.get("source") or "unknown"
        if not q or not a:
            continue
        qk = q.lower()
        ak = a[:160].strip().lower()
        audit_stream.append({"q": q, "a": a, "source": src})
        cur = by_q.get(qk)
        if cur is None or pref_rank.get(src,0) > pref_rank.get(cur["source"],0):
            by_q[qk] = {"q": q, "a": a, "source": src}
    for qk, qa in by_q.items():
        ak = qa["a"][:160].strip().lower()
        if qk in seen_q:
            dup_q += 1
            continue
        if ak in seen_a:
            dup_a += 1
            continue
        seen_q[qk] = True
        seen_a[ak] = True
        merged.append(qa)
    return merged, src_counts, dup_q, dup_a, audit_stream

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

def _audit_single_qa(qa: Dict[str,str]) -> Dict[str, Any]:
    q0 = qa.get("q") or ""
    a0 = qa.get("a") or ""
    issues: List[str] = []
    fixes: Dict[str,str] = {}
    q_norm = _normalize_question(_strip_html(q0))
    if q_norm != q0:
        issues.append("question_format")
        fixes["q"] = q_norm
    a_clean = _clean_answer(a0)
    if a_clean != a0:
        issues.append("answer_html_removed")
    if _has_marketing_filler(a_clean):
        issues.append("marketing_filler")
    first = _first_sentence(a_clean)
    a_first = first if first else a_clean
    if len(a_first.split()) > 80:
        a_prop, _ = _trim_words(a_first, 80)
        issues.append("answer_over_80w")
        fixes["a"] = a_prop
    else:
        a_prop = a_first
    status = "ok" if not issues else "fix"
    return {
        "status": status,
        "issues": issues,
        "proposed": {"q": fixes.get("q", q_norm), "a": fixes.get("a", a_prop)}
    }

def _parity_ok(jsonld_qas: List[Dict[str,str]], visible_qas: List[Dict[str,str]]) -> bool:
    if not jsonld_qas or not visible_qas:
        return True
    jq = set(q["q"].strip().lower() for q in jsonld_qas if q.get("q"))
    vq = set(q["q"].strip().lower() for q in visible_qas if q.get("q"))
    if not jq or not vq:
        return True
    inter = len(jq & vq)
    union = len(jq | vq)
    jacc = inter / max(1, union)
    return jacc >= 0.4

def _has_faq_schema_flag(page: Dict[str, Any], faq_jsonld: Any, schema_flags: Dict[str,bool], url: str) -> bool:
    types = [str(t).lower() for t in (page.get("jsonld_types") or [])]
    if "faqpage" in types:
        return True
    if faq_jsonld:
        return True
    if schema_flags.get(url):
        return True
    return False

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
    if qas > 0 and not has_faq_schema and ptype == "faq":
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

# -------- main job --------
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
        if isinstance(faq_jsonld_raw, str) and faq_jsonld_raw.strip():
            try:
                faq_jsonld = json.loads(faq_jsonld_raw)
            except Exception:
                faq_jsonld = []
        else:
            faq_jsonld = faq_jsonld_raw if isinstance(faq_jsonld_raw, (list, dict)) else []

        q_jsonld = _qas_from_jsonld(faq_jsonld)
        q_visible = _extract_visible_from_page(p)
        q_faqjob = faq_by_url.get(url, [])

        merged, src_counts, dup_q, dup_a, audit_stream = _merge_qas(q_jsonld, q_faqjob, q_visible)
        gt80, le80 = _answers_count(merged)
        has_faq_schema = _has_faq_schema_flag(p, faq_jsonld, faq_schema_flags, url)
        parity_ok = _parity_ok(q_jsonld, q_visible)

        counts = {
            "merged": len(merged),
            "answers_gt_80w": gt80,
            "answers_leq_80w": le80
        }
        score, issues = _score_and_issues(ptype, counts, has_faq_schema, dup_q, dup_a, parity_ok)

        limited_qas = _limit_qas(merged, qas_per_page)

        qas_ready: List[Dict[str, str]] = []
        qa_audit_items: List[Dict[str, Any]] = []
        ok_count = 0
        fix_count = 0
        for qa in limited_qas:
            audit = _audit_single_qa(qa)
            qa_audit_items.append({
                "source": qa.get("source"),
                "q_orig": qa.get("q"),
                "a_orig": qa.get("a"),
                "status": audit["status"],
                "issues": audit["issues"],
                "proposed_q": audit["proposed"]["q"],
                "proposed_a": audit["proposed"]["a"]
            })
            if audit["status"] == "ok":
                ok_count += 1
            else:
                fix_count += 1
            qas_ready.append({"q": audit["proposed"]["q"], "a": audit["proposed"]["a"]})

        metrics = {
            "qas_detected": len(merged),
            "answers_gt_80w": gt80,
            "answers_leq_80w": le80,
            "has_faq_schema_detected": bool(has_faq_schema),
            "src_counts": src_counts,
            "dup_questions": dup_q,
            "dup_answers": dup_a,
            "parity_ok": parity_ok,
            "qa_ok": ok_count,
            "qa_need_fixes": fix_count
        }

        out_pages.append({
            "url": url,
            "lang": lang,
            "type": ptype,
            "score": score,
            "issues": issues,
            "metrics": metrics,
            "qas": qas_ready,
            "qa_audit": qa_audit_items,
            "faq_jsonld_present": bool(faq_jsonld)
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
