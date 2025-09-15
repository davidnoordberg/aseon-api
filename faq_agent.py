# faq_agent.py
import os
import json
import re
from typing import Dict, Any, List, Optional
from openai import OpenAI
from psycopg.rows import dict_row

from rag_helper import search_site_docs, search_kb, build_context

CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "20"))
MAX_CTX_CHARS = int(os.getenv("FAQ_MAX_CTX_CHARS", "3500"))

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _cap_words(s: str, max_words: int) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    if not s:
        return s
    words = s.split(" ")
    return " ".join(words[:max_words])


def _clean_url(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    u = u.strip()
    if u.startswith("http://") or u.startswith("https://"):
        return u
    return None


def _fallback_crawl_snapshot(conn, site_id: str, max_pages: int = 6) -> str:
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT output
                  FROM jobs
                 WHERE site_id=%s AND type='crawl' AND status='done'
              ORDER BY COALESCE(finished_at, created_at) DESC
                 LIMIT 1
            """,
                (site_id,),
            )
            r = cur.fetchone()
        out = (r or {}).get("output") or {}
        pages = out.get("pages") or []
        bits: List[str] = []
        for p in pages[:max_pages]:
            url = p.get("final_url") or p.get("url") or ""
            title = (p.get("title") or "")[:200]
            h1 = (p.get("h1") or "")[:200]
            meta = (p.get("meta_description") or "")[:300]
            paras = " ".join((p.get("paragraphs") or [])[:2])[:600]
            snippet = "\n".join([x for x in [url, title, h1, meta, paras] if x])
            if snippet.strip():
                bits.append(snippet)
        return "\n".join(bits).strip()
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        print(
            json.dumps(
                {"level": "WARN", "msg": "faq_crawl_ctx_failed", "error": str(e)[:300]}
            ),
            flush=True,
        )
        return ""


def _map_id_to_url(raw: Optional[str], site_cites: List[dict], kb_cites: List[dict]) -> Optional[str]:
    """
    Accepts things like 'S1', '[S2]', 'K3' and returns the mapped URL if present.
    """
    if not raw:
        return None
    s = raw.strip()
    m = re.match(r"^\[?\s*([SK])\s*(\d+)\s*\]?$", s, flags=re.I)
    if not m:
        return None
    kind = m.group(1).upper()
    idx = m.group(2)
    target_id = f"{kind}{idx}"
    if kind == "S":
        for c in site_cites or []:
            if (c.get("id") or "").upper() == target_id:
                return _clean_url(c.get("url"))
    else:
        for c in kb_cites or []:
            if (c.get("id") or "").upper() == target_id:
                return _clean_url(c.get("url"))
    return None


def _normalize_source(
    src: Optional[str],
    ctx: Dict[str, Any],
) -> Optional[str]:
    """
    Normalize the 'source' field to a URL if possible:
    - If it's already a URL with http(s) → keep.
    - If it's S#/K# → map via citations.
    - Else pick the first site citation URL, if any; otherwise None.
    """
    # URL already?
    url = _clean_url(src)
    if url:
        return url

    # ID mapping (S#/K#)
    mapped = _map_id_to_url(src, ctx.get("site_citations") or [], ctx.get("kb_citations") or [])
    if mapped:
        return mapped

    # Fallback: first site citation if available
    cites = ctx.get("site_citations") or []
    if cites:
        return _clean_url(cites[0].get("url"))

    return None


def generate_faqs(conn, site_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    topic = (payload or {}).get("topic") or "general"
    count = int((payload or {}).get("count", 6))
    max_words = int((payload or {}).get("max_words", 80))
    kb_tags = (payload or {}).get("kb_tags") or ["Schema", "SEO", "AEO", "Content", "Quality"]
    use_context = (payload or {}).get("use_context", "auto")

    site_rows: List[dict] = []
    kb_rows: List[dict] = []
    crawl_ctx = ""

    if use_context in ("auto", "documents", "kb", "crawl"):
        if use_context in ("auto", "documents"):
            site_rows = search_site_docs(conn, site_id, topic, k=8)
        if use_context in ("auto", "kb"):
            kb_rows = search_kb(conn, topic, k=6, tags=kb_tags)
        if use_context in ("auto", "crawl"):
            crawl_ctx = _fallback_crawl_snapshot(conn, site_id, max_pages=6)

    ctx = build_context(site_rows, kb_rows, budget_chars=MAX_CTX_CHARS)

    full_site_ctx = (ctx.get("site_ctx") or "")
    if crawl_ctx:
        extra = "\n".join([f"[S*] Crawl Snapshot\n{crawl_ctx}"])
        full_site_ctx = (full_site_ctx + "\n" + extra).strip()

    system = (
        f"You write concise, factual FAQs grounded ONLY in the provided context.\n"
        f"Return EXACTLY {count} Q/A pairs. Each answer ≤ {max_words} words. JSON only.\n"
        "Rules:\n"
        "- Prefer facts from [S#] site snippets; fall back to [K#] policy/best practices.\n"
        '- Every FAQ must include a "source": ONE best URL (prefer site). If nothing supports it, use null.\n'
        '- No fluff or speculation; if unknown from context, state "Cannot answer from context.".\n'
        'Return:\n{"faqs":[{"q":"...","a":"...","source":"https://...|null"}]}'
    )

    user = f"""Topic: {topic}

--- SITE CONTEXT ---
{full_site_ctx}

--- KB CONTEXT ---
{ctx.get("kb_ctx") or ""}
"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        timeout=OPENAI_TIMEOUT_SEC,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )

    faqs: List[Dict[str, Any]] = []
    try:
        data = json.loads(resp.choices[0].message.content)
        raw = data.get("faqs") or []
    except Exception:
        raw = []

    # Deduplicate by question (case-insensitive)
    seen_q = set()
    for f in raw[: count + 2]:  # take a little extra, we will trim to 'count'
        q = (f.get("q") or "").strip()
        if not q:
            continue
        key = q.lower()
        if key in seen_q:
            continue
        seen_q.add(key)
        a = _cap_words((f.get("a") or ""), max_words)
        src = _normalize_source(f.get("source"), ctx)
        if q and a:
            faqs.append({"q": q, "a": a if a else "Cannot answer from context.", "source": src})

        if len(faqs) >= count:
            break

    # Pad if model under-delivered
    if len(faqs) < count:
        for i in range(len(faqs), count):
            faqs.append(
                {
                    "q": f"{topic.capitalize()} FAQ {i+1}",
                    "a": _cap_words("Cannot answer from context.", max_words),
                    "source": None,
                }
            )

    out = {
        "_context_used": {
            "site_citations": ctx.get("site_citations"),
            "kb_citations": ctx.get("kb_citations"),
            "char_used": ctx.get("char_used"),
            "used": [
                x
                for x in [
                    "documents" if site_rows else None,
                    "kb" if kb_rows else None,
                    "crawl" if crawl_ctx else None,
                ]
                if x
            ],
        },
        "faqs": faqs[:count],
    }

    print(
        json.dumps(
            {
                "level": "INFO",
                "msg": "faq_generated",
                "topic": topic,
                "n": len(out["faqs"]),
                "used": out["_context_used"]["used"],
            }
        ),
        flush=True,
    )
    return out
