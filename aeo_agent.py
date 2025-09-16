import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

from openai import OpenAI
from psycopg.rows import dict_row

OPENAI_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "30"))
_openai_key = os.getenv("OPENAI_API_KEY")
_openai_client = OpenAI(api_key=_openai_key) if _openai_key else None


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


def _autolang(site_meta: Dict[str, Any], crawl: Optional[Dict[str, Any]]) -> str:
    lang = (site_meta.get("language") or "").strip().lower()
    if lang:
        return "nl" if lang.startswith("nl") else "en"
    c = (site_meta.get("country") or "").strip().lower()
    if c in ("nl", "nld", "netherlands"):
        return "nl"
    return "en"


def _page_language(page: Dict[str, Any], default_lang: str) -> str:
    return (page.get("language") or default_lang).split("-")[0].lower()


def _word_count(text: str) -> int:
    return len((text or "").strip().split())


def _has_faq_schema(page: Dict[str, Any]) -> bool:
    types = [t.lower() for t in (page.get("jsonld_types") or [])]
    return any(t == "faqpage" for t in types)


def _shorten(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[: n - 1] + "…")


def _llm_qas_from_page(lang: str, title: str, h1: str, body_preview: str, kb_ctx: str, n: int = 4) -> List[Dict[str, str]]:
    if not _openai_client:
        topic = (h1 or title or "").strip() or "deze pagina"
        if lang == "nl":
            qs = [
                {"q": f"Wat is {topic.lower()}?", "a": f"{topic} in één alinea uitgelegd in eenvoudige taal."},
                {"q": f"Hoe werkt {topic.lower()}?", "a": "Korte uitleg in 40–60 woorden, met 2–3 concrete stappen."},
                {"q": f"Wat levert {topic.lower()} op?", "a": "Resultaten/voordelen in 40–60 woorden, met 1 meetbaar voorbeeld."},
                {"q": f"Wat zijn de eerste stappen?", "a": "Genummerde mini-stappen (3–5) in 50–70 woorden."},
            ]
        else:
            qs = [
                {"q": f"What is {topic}?", "a": f"{topic} explained in a single paragraph of plain language."},
                {"q": "How does it work?", "a": "40–60 words with 2–3 concrete steps."},
                {"q": "What are the benefits?", "a": "40–60 words with one measurable example."},
                {"q": "How do I get started?", "a": "3–5 numbered mini-steps in 50–70 words."},
            ]
        return qs[:n]

    sys = (
        "You create snippet-ready Q&A for Answer/Generative Engines. "
        "Write in the requested language. Each answer must be ≤80 words, lead with the answer, "
        "use only facts from the provided context."
    )
    user = {
        "language": lang,
        "title": title,
        "h1": h1,
        "body_preview": _shorten(body_preview, 1200),
        "kb": _shorten(kb_ctx, 1500),
        "n": n,
    }
    try:
        resp = _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            timeout=OPENAI_TIMEOUT_SEC,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
        )
        txt = resp.choices[0].message.content.strip()
        m = re.search(r"\[.*\]", txt, re.S)
        data = json.loads(m.group(0)) if m else []
        out = []
        for x in data:
            q = (x.get("q") or x.get("question") or "").strip()
            a = (x.get("a") or x.get("answer") or "").strip()
            if q and a:
                out.append({"q": q, "a": a})
        return out[:n]
    except Exception:
        return []


def _faq_html_block(qas: List[Dict[str, str]], lang: str) -> str:
    label = "Veelgestelde vragen" if lang == "nl" else "Frequently asked questions"
    lis = []
    for qa in qas:
        q = qa["q"].strip()
        a = qa["a"].strip()
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
                "name": qa["q"],
                "acceptedAnswer": {"@type": "Answer", "text": qa["a"]},
            }
        )
    return {"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": items}


def _kb_context_snippet() -> str:
    return ""


def _score_page(has_faq_schema: bool, qas: List[Dict[str, str]]) -> Tuple[int, Dict[str, Any], List[str]]:
    answers_ok = sum(1 for x in qas if _word_count(x["a"]) <= 80)
    n = len(qas)
    metrics = {"qas": n, "answers_leq_80w": answers_ok, "has_faq_schema": has_faq_schema}
    issues: List[str] = []
    if n < 3:
        issues.append("Too few Q&A (min 3).")
    if answers_ok < n:
        issues.append("Some answers >80 words.")
    if not has_faq_schema:
        issues.append("No FAQPage JSON-LD.")
    score = 0
    score += min(n, 6) * 10
    score += answers_ok * 5
    score += 10 if has_faq_schema else 0
    score = min(score, 100)
    return score, metrics, issues


def _unique_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bucket: Dict[str, Dict[str, Any]] = {}

    def _score(pp: Dict[str, Any]) -> Tuple[int, int, int]:
        status = int(pp.get("status") or 0)
        is200 = 1 if status == 200 else 0
        wc = int(pp.get("word_count") or 0)
        txt_len = len((pp.get("text") or pp.get("excerpt") or "")[:2000])
        return (is200, wc, txt_len)

    for p in pages or []:
        u = _norm_url(p.get("final_url") or p.get("url") or "")
        if not u:
            continue
        cur = bucket.get(u)
        if (cur is None) or (_score(p) > _score(cur)):
            # also ensure final_url/url normalized in the kept record
            p2 = dict(p)
            p2["final_url"] = u
            p2["url"] = u
            bucket[u] = p2
    return [v for _, v in bucket.items()]


def generate_aeo(conn, job):
    site_id = job["site_id"]
    payload = job.get("payload") or {}
    per_page_qas = max(1, int(payload.get("per_page_qas") or 4))
    use_kb = bool(payload.get("use_kb", True))
    only_status_200 = bool(payload.get("only_status_200", False))

    site_meta = _fetch_site_meta(conn, site_id)
    crawl = _fetch_latest_job(conn, site_id, "crawl") or {}
    faq_job = _fetch_latest_job(conn, site_id, "faq") or {}

    lang_default = _autolang(site_meta, crawl)
    kb_ctx = _kb_context_snippet() if use_kb else ""

    src_pages = _unique_pages(crawl.get("pages") or [])
    pages: List[Dict[str, Any]] = []
    seen: Dict[str, bool] = {}

    for p in src_pages:
        url = _norm_url(p.get("final_url") or p.get("url") or "")
        if not url or seen.get(url):
            continue
        if only_status_200 and int(p.get("status") or 0) != 200:
            continue
        seen[url] = True

        title = (p.get("title") or "").strip()
        h1 = (p.get("h1") or "").strip()
        page_lang = _page_language(p, lang_default)
        body_preview = (p.get("text") or p.get("excerpt") or "")

        reused: List[Dict[str, str]] = []
        for item in (faq_job.get("faqs") or []):
            src = (item.get("source") or "").strip()
            if src and src.startswith(url):
                q = (item.get("q") or "").strip()
                a = (item.get("a") or "").strip()
                if q and a:
                    reused.append({"q": q, "a": a})

        need = max(0, per_page_qas - len(reused))
        extra = _llm_qas_from_page(page_lang, title, h1, body_preview, kb_ctx, n=need) if need else []
        qas = (reused + extra)[:per_page_qas]

        faq_html = _faq_html_block(qas, page_lang)
        faq_jsonld = _faq_jsonld(qas)
        has_schema = _has_faq_schema(p)
        score, metrics, issues = _score_page(has_schema, qas)

        pages.append(
            {
                "url": url,
                "lang": page_lang,
                "title": title,
                "h1": h1,
                "qas": qas,
                "score": score,
                "metrics": metrics,
                "issues": issues,
                "faq_html": faq_html,
                "faq_jsonld": faq_jsonld,
            }
        )

    avg = round(sum(x["score"] for x in pages) / max(1, len(pages)))
    needs = sum(1 for x in pages if not x["metrics"]["has_faq_schema"])
    summary = {"pages_total": len(pages), "avg_score": avg, "pages_missing_faq_schema": needs}

    return {
        "site": {"url": site_meta.get("url"), "language": lang_default, "account_name": site_meta.get("account_name")},
        "summary": summary,
        "pages": pages,
    }
