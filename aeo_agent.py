# aeo_agent.py — deterministic AEO builder with strict FAQ gating + ASCII-safe answers
# Public API: generate_aeo(conn, job) -> dict with {"site":{}, "pages":[...]}
# Emits Q&A only for pages that are likely FAQ (URL contains /faq) or have FAQPage JSON-LD.

import os
import re
import json
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urlsplit
from html import unescape as html_unescape
from psycopg.rows import dict_row

QUESTION_PREFIXES = (
    "what ", "how ", "why ", "when ", "can ", "does ", "do ", "is ", "are ", "should ", "will ", "where ", "who ",
    "wat ", "hoe ", "waarom ", "wanneer ", "kan ", "doet ", "doen ", "is ", "zijn ", "moet ", "zal ", "waar ", "wie "
)

_TAG_RE = re.compile(r"<[^>]+>")

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
        row = cur.fetchone() or {}
        url = (row.get("url") or "").strip()
        if url:
            # normalize to www if naked domain is used
            if url.startswith("https://aseon.io") or url.startswith("http://aseon.io"):
                url = url.replace("aseon.io", "www.aseon.io")
            if not url.endswith("/"):
                url += "/"
            row["url"] = url
        return row

def _fetch_latest_job(conn, site_id: str, jtype: str) -> Optional[Dict[str, Any]]:
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

def _is_likely_faq_url(u: str) -> bool:
    try:
        p = urlsplit(u)
        path = (p.path or "").lower()
        frag = (p.fragment or "").lower()
        s = path + ("#" + frag if frag else "")
        return "/faq" in s.split("?")[0]
    except Exception:
        return False

def _strip_html(s: str) -> str:
    if not s: return ""
    s = html_unescape(str(s))
    s = _TAG_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _normalize_punct(s: str) -> str:
    t = (s or "")
    repl = {
        "\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00A0": " "
    }
    for k,v in repl.items():
        t = t.replace(k, v)
    # collapse repeated hyphens that might come from em dashes
    t = re.sub(r"-{2,}", "-", t)
    return t

def _looks_like_question(text: str) -> bool:
    if not text: return False
    t = text.strip()
    low = t.lower()
    if "?" in t: return True
    if any(low.startswith(p) for p in QUESTION_PREFIXES) and len(low.split()) >= 2: return True
    if re.match(r"^(q|vraag)\s*[:\-–]\s+\S", low): return True
    return False

def _normalize_question(q: str) -> str:
    t = _normalize_punct(_strip_html(q))
    t = re.sub(r"\s+", " ", t).strip()
    if t and not t.endswith("?"):
        if any(t.lower().startswith(p) for p in QUESTION_PREFIXES):
            t = t.rstrip(".! ") + "?"
    if t:
        t = t[0].upper() + t[1:]
    return t

def _clean_answer(a: str) -> str:
    a0 = _normalize_punct(_strip_html(a))
    a0 = re.sub(r"\s+", " ", a0).strip()
    return a0

def _sentences(s: str) -> List[str]:
    s = (s or "").strip()
    parts = re.split(r"(?<=[.!?])\s+", s)
    return [p.strip() for p in parts if p.strip()]

def _trim_words(s: str, limit: int = 80) -> Tuple[str, bool]:
    words = (s or "").split()
    if len(words) <= limit:
        return " ".join(words), False
    return " ".join(words[:limit]) + "…", True

def _has_marketing_filler(s: str) -> bool:
    t = (s or "").lower()
    bad = ["contact us","get in touch","learn more","click here","our team","we offer","we provide","ask us"]
    return any(b in t for b in bad)

def _improve_answer(a: str) -> str:
    a = _clean_answer(a)
    if _has_marketing_filler(a):
        a = re.split(r"(?:contact|learn more|get in touch)[:.!?]", a, maxsplit=1, flags=re.I)[0].strip() or a
    sents = _sentences(a)
    if sents:
        out = []
        for s in sents:
            out.append(s)
            if len(" ".join(out).split()) >= 18:
                break
        text = " ".join(out).strip()
    else:
        text = a
    text, _ = _trim_words(text, 80)
    return text

def _qas_from_jsonld(faq_jsonld_any: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []

    def _handle_entity(entity):
        if not isinstance(entity, dict): return
        q = entity.get("name") or entity.get("question") or entity.get("headline") or ""
        ans = entity.get("acceptedAnswer") or entity.get("accepted_answer") or {}
        if isinstance(ans, list) and ans:
            ans = ans[0]
        a_text = ""
        if isinstance(ans, dict):
            a_text = ans.get("text") or ans.get("answer") or ans.get("articleBody") or ""
        q = _normalize_question(q)
        a_text = _clean_answer(a_text)
        if q and a_text:
            out.append({"q": q, "a": a_text, "source": "jsonld"})

    def _handle_root(obj):
        if not isinstance(obj, dict): return
        if str(obj.get("@type","")).lower() == "faqpage":
            main = obj.get("mainEntity") or obj.get("main_entity") or []
            if isinstance(main, list):
                for ent in main: _handle_entity(ent)
            else:
                _handle_entity(main)
        if "@graph" in obj and isinstance(obj["@graph"], list):
            for node in obj["@graph"]:
                _handle_root(node)

    if isinstance(faq_jsonld_any, list):
        for x in faq_jsonld_any: _handle_root(x)
    elif isinstance(faq_jsonld_any, dict):
        _handle_root(faq_jsonld_any)
    return _dedupe_qas(out)

def _extract_visible_from_page(p: Dict[str, Any]) -> List[Dict[str, str]]:
    qas = []
    for qa in (p.get("faq_visible") or []):
        q = _normalize_question(qa.get("q") or "")
        a = _clean_answer(qa.get("a") or "")
        if q and a:
            qas.append({"q": q, "a": a, "source": "visible"})
    return _dedupe_qas(qas)

def _dedupe_qas(qas: List[Dict[str,str]]) -> List[Dict[str,str]]:
    seen = set()
    out: List[Dict[str,str]] = []
    for qa in qas:
        q = (qa.get("q") or "").strip()
        a = (qa.get("a") or "").strip()
        if not q or not a: continue
        k = q.lower()
        if k in seen: continue
        seen.add(k)
        out.append({"q": q, "a": a, "source": qa.get("source")})
    return out

def _merge_qas(q_jsonld: List[Dict[str, str]], q_visible: List[Dict[str, str]]):
    all_src = q_visible + q_jsonld
    src_counts = {"jsonld": len(q_jsonld), "visible": len(q_visible), "faq_job": 0}
    by_q: Dict[str, Dict[str,str]] = {}
    seen_a = {}
    dup_q = 0
    dup_a = 0
    for qa in all_src:
        q = (qa.get("q") or "").strip()
        a = (qa.get("a") or "").strip()
        src = qa.get("source") or ""
        if not q or not a: continue
        qk = q.lower()
        if qk in by_q:
            dup_q += 1
            continue
        by_q[qk] = {"q": q, "a": a, "source": src}
        ak = a[:160].lower()
        if ak in seen_a:
            dup_a += 1
        else:
            seen_a[ak] = 1
    merged = list(by_q.values())
    gt80 = sum(1 for qa in merged if len((qa.get("a") or "").split()) > 80)
    le80 = len(merged) - gt80
    return merged, src_counts, dup_q, dup_a, gt80, le80

def _has_faq_schema_flag(page: Dict[str, Any], faq_jsonld: Any) -> bool:
    types = [str(t).lower() for t in (page.get("jsonld_types") or [])]
    if "faqpage" in types: return True
    if faq_jsonld: return True
    return False

def _infer_ptype(url: str, faq_jsonld_any: Any) -> str:
    if _is_likely_faq_url(url): return "faq"
    if faq_jsonld_any: return "faq"
    return "other"

def _score_and_issues(ptype: str, qas_len: int, gt80: int, has_schema: bool, dup_q: int, dup_a: int, parity_ok: bool) -> Tuple[int, List[str]]:
    issues: List[str] = []
    score = 100
    if qas_len == 0:
        issues.append("No Q&A section on page.")
        score -= 85
    if ptype == "faq" and 0 < qas_len < 3:
        issues.append("Too few Q&A (min 3).")
        score -= 20
    if ptype == "faq" and qas_len > 0 and not has_schema:
        issues.append("No FAQPage JSON-LD.")
        score -= 15
    if gt80 > 0:
        issues.append("Overlong answers (>80 words).")
        score -= min(20, gt80 * 5)
    if dup_q > 0:
        issues.append("Duplicate questions.")
        score -= min(10, dup_q * 2)
    if dup_a > 0:
        issues.append("Duplicate answers across sources.")
        score -= min(10, dup_a * 2)
    if qas_len > 0 and not parity_ok:
        issues.append("Schema/text parity mismatch.")
        score -= 10
    score = max(0, min(100, score))
    return score, issues

def generate_aeo(conn, job):
    site_id = job["site_id"]
    payload = job.get("payload") or {}
    qas_per_page = int(payload.get("qas_per_page") or 40)

    site_meta = _fetch_site_meta(conn, site_id)
    crawl = _fetch_latest_job(conn, site_id, "crawl")
    if not crawl:
        raise ValueError("no_crawl_data")

    out_pages: List[Dict[str, Any]] = []

    for p in (crawl.get("pages") or []):
        url = (p.get("url") or "").strip()
        if not url:
            continue

        # Normalize/parse FAQ JSON-LD
        faq_jsonld_raw = p.get("faq_jsonld")
        faq_jsonld = None
        if isinstance(faq_jsonld_raw, str) and faq_jsonld_raw.strip():
            try:
                faq_jsonld = json.loads(faq_jsonld_raw)
            except Exception:
                faq_jsonld = None
        elif isinstance(faq_jsonld_raw, (list, dict)):
            faq_jsonld = faq_jsonld_raw

        # Collect sources
        q_jsonld = _qas_from_jsonld(faq_jsonld) if faq_jsonld else []
        q_visible = _extract_visible_from_page(p)

        merged, src_counts, dup_q, dup_a, gt80, le80 = _merge_qas(q_jsonld, q_visible)
        parity_ok = True
        if q_jsonld and q_visible:
            jq = set(x["q"].lower() for x in q_jsonld)
            vq = set(x["q"].lower() for x in q_visible)
            inter = len(jq & vq); union = len(jq | vq) or 1
            parity_ok = (inter / union) >= 0.4

        has_schema = _has_faq_schema_flag(p, faq_jsonld)
        ptype = _infer_ptype(url, faq_jsonld)

        # Build snippet-ready answers (ASCII-safe)
        snippet_qas: List[Dict[str,str]] = []
        if ptype == "faq":
            for qa in merged[:qas_per_page]:
                q = _normalize_question(qa.get("q") or "")
                a = _improve_answer(qa.get("a") or "")
                snippet_qas.append({"q": q, "a": a})

        score, issues = _score_and_issues(ptype, len(merged), gt80, has_schema, dup_q, dup_a, parity_ok)

        out_pages.append({
            "url": url,
            "type": ptype,
            "score": score,
            "issues": issues,
            "metrics": {
                "src_counts": src_counts,
                "qas_detected": len(merged),
                "answers_gt_80w": gt80,
                "answers_leq_80w": le80,
                "dup_questions": dup_q,
                "dup_answers": dup_a,
                "qa_ok": sum(1 for qa in merged if len((qa.get("a") or "").split()) <= 80),
                "qa_need_fixes": gt80,
                "parity_ok": parity_ok,
                "has_faq_schema_detected": bool(has_schema),
            },
            "qas": snippet_qas,
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
