# faq_agent.py
# Genereert FAQ's met voorkeur voor RAG uit 'documents' (pgvector).
# Valt terug op crawl-context of, als dat er niet is, op model-only.
from __future__ import annotations
import os, json, math, textwrap
from typing import List, Dict, Any, Optional, Tuple

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# Retrieval settings
TOP_K = int(os.getenv("FAQ_TOP_K", "5"))
MAX_SOURCES_PER_Q = int(os.getenv("FAQ_MAX_SOURCES_PER_Q", "1"))  # we nemen 1 beste bron-URL
LIKE_LIMIT = int(os.getenv("FAQ_LIKE_LIMIT", "8"))  # fallback simple search limiet

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ---------- util ----------

def _log(level: str, msg: str, **kw):
    try:
        print(json.dumps({"level": level.upper(), "msg": msg, **kw}), flush=True)
    except Exception:
        # nooit hard crashen op log
        pass

def _trim_words(s: str, max_words: int) -> str:
    if not s or max_words <= 0:
        return s or ""
    words = s.strip().split()
    if len(words) <= max_words:
        return s.strip()
    return " ".join(words[:max_words])

def _embed_texts(texts: List[str]) -> List[List[float]]:
    # batched embeddings om latency te beperken (kleine batches volstaan prima)
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def _embed_one(text: str) -> List[float]:
    return _embed_texts([text])[0]

# ---------- context fetchers ----------

def _documents_exist(conn, site_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM documents WHERE site_id=%s LIMIT 1", (site_id,))
        return cur.fetchone() is not None

def _latest_crawl_context(conn, site_id: str, max_chars: int = 2400) -> Optional[str]:
    """Pak een compacte context uit de meest recente crawl-job (title/h1/h2/h3)."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT output
              FROM jobs
             WHERE site_id=%s AND type='crawl' AND status='done' AND output IS NOT NULL
          ORDER BY finished_at DESC NULLS LAST, created_at DESC
             LIMIT 1
        """, (site_id,))
        row = cur.fetchone()
        if not row or not row.get("output"):
            return None
        out = row["output"] or {}
        pages = out.get("pages") or []
        bits: List[str] = []
        for p in pages[:12]:  # eerste dozen pagina’s is vaak genoeg
            for k in ("title", "h1"):
                v = p.get(k)
                if v: bits.append(str(v))
            for h in (p.get("h2") or []):
                bits.append(str(h))
            for h in (p.get("h3") or []):
                bits.append(str(h))
        if not bits:
            return None
        s = "\n".join(bits).strip()
        return s[:max_chars]

def _vector_sources_for_question(conn, site_id: str, question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Zoek relevante documenten voor de vraag via pgvector. Return [{url, content}, ...]."""
    try:
        qvec = _embed_one(question)
    except Exception as e:
        _log("WARN", "faq_embed_failed", error=str(e))
        return []

    with conn.cursor(row_factory=dict_row) as cur:
        try:
            # Belangrijk: %s::vector cast voor de query-embedding.
            cur.execute(
                """
                SELECT url, content
                  FROM documents
                 WHERE site_id = %s
              ORDER BY embedding <-> %s::vector
                 LIMIT %s
                """,
                (site_id, qvec, top_k),
            )
            rows = cur.fetchall() or []
            return [{"url": r["url"], "content": r["content"]} for r in rows]
        except Exception as e:
            _log("WARN", "faq_vector_search_failed", error=str(e))
            return []

def _like_sources_for_question(conn, site_id: str, question: str, limit: int = LIKE_LIMIT) -> List[Dict[str, Any]]:
    """Fallback LIKE-zoek: simpele tekstmatch in documents.content."""
    with conn.cursor(row_factory=dict_row) as cur:
        try:
            cur.execute(
                """
                SELECT url, content
                  FROM documents
                 WHERE site_id = %s
                   AND content ILIKE %s
                 LIMIT %s
                """,
                (site_id, f"%{question.split()[0]}%", limit),
            )
            rows = cur.fetchall() or []
            return [{"url": r["url"], "content": r["content"]} for r in rows]
        except Exception as e:
            _log("ERROR", "faq_like_search_failed", error=str(e))
            return []

# ---------- LLM prompt ----------

_SYS = """You write concise, factual FAQs for websites. Answers must be short, neutral in tone, and avoid marketing claims.
Rules:
- Return ONLY a JSON object: {"faqs":[{"q":"...","a":"..."}]}
- 5–10 Q&A pairs unless a specific count is requested.
- Max words per answer = max_words (provided).
- Avoid hallucinations: if context is present, stick to it.
- Avoid over-claiming; prefer objective phrasing.
"""

def _build_user_prompt(topic: str, max_words: int, context: Optional[str]) -> str:
    base = {
        "topic": topic,
        "max_words": max_words,
        "note": "Keep Qs useful and distinct. Prefer how/what/why questions."
    }
    if context:
        return "Context:\n" + context + "\n\nSpec:\n" + json.dumps(base, ensure_ascii=False)
    return "No extra context.\nSpec:\n" + json.dumps(base, ensure_ascii=False)

def _call_llm_faq(topic: str, max_words: int, context: Optional[str], count: int) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content": _SYS},
        {"role": "user", "content": _build_user_prompt(topic, max_words, context)},
    ]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        faqs = data.get("faqs") or []
        # trim naar requested count (als model eroverheen ging)
        return faqs[:count]
    except Exception as e:
        _log("WARN", "faq_parse_failed", error=str(e))
        return []

# ---------- hoofd-API ----------

def generate_faqs(conn, site_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload:
      topic: str
      count: int (default 6)
      use_context: "auto" | "documents" | "crawl" | "none"
      max_words: int (default 80)
    return:
      {"_context_used": "...", "faqs":[{"q","a","source"}], "topic": "..."}
    """
    topic = (payload or {}).get("topic") or "general"
    count = int((payload or {}).get("count", 6))
    max_words = int((payload or {}).get("max_words", 80))
    use_ctx = (payload or {}).get("use_context", "auto").lower()

    # 1) Bepaal context
    ctx_used = "none"
    context_blob: Optional[str] = None

    if use_ctx in ("auto", "documents"):
        if _documents_exist(conn, site_id):
            ctx_used = "documents"
        elif use_ctx == "documents":
            ctx_used = "none"
        elif use_ctx == "auto":
            # probeer crawl als documents er (nog) niet zijn
            crawl_ctx = _latest_crawl_context(conn, site_id)
            if crawl_ctx:
                ctx_used = "crawl"
                context_blob = crawl_ctx
            else:
                ctx_used = "none"
    if use_ctx == "crawl" and ctx_used == "none":
        crawl_ctx = _latest_crawl_context(conn, site_id)
        if crawl_ctx:
            ctx_used = "crawl"
            context_blob = crawl_ctx

    # Als documents als context gekozen zijn, hoeven we geen groot tekstblob op te bouwen;
    # we gebruiken retrieval per vraag om een bron-URL te selecteren. Voor het genereren
    # geven we eventueel een korte "site focus" uit crawl mee (optioneel).
    if ctx_used == "documents" and context_blob is None:
        # kleine hint-context kan helpen, maar is optioneel
        context_blob = _latest_crawl_context(conn, site_id, max_chars=1200) or None

    # 2) Genereer FAQ’s
    faqs = _call_llm_faq(topic=topic, max_words=max_words, context=context_blob, count=count)

    # 3) Post-process: trim answers en koppel een bron-URL via retrieval
    out_faqs: List[Dict[str, Any]] = []
    for qa in faqs:
        q = (qa.get("q") or "").strip()
        a = (qa.get("a") or "").strip()
        if not q or not a:
            continue

        a = _trim_words(a, max_words)

        source_url: Optional[str] = None
        if ctx_used == "documents":
            # vector retrieval
            srcs = _vector_sources_for_question(conn, site_id, q, top_k=TOP_K)
            if not srcs:
                # fallback LIKE
                srcs = _like_sources_for_question(conn, site_id, q, limit=LIKE_LIMIT)
            if srcs:
                source_url = srcs[0]["url"]
        elif ctx_used == "crawl":
            # eenvoudige bron: probeer te raden door LIKE op documents als die tóch bestaan,
            # anders laat source leeg (crawl-output zit niet in DB als documents ontbreekt).
            if _documents_exist(conn, site_id):
                srcs = _like_sources_for_question(conn, site_id, q, limit=LIKE_LIMIT)
                if srcs:
                    source_url = srcs[0]["url"]

        out_faqs.append({"q": q, "a": a, "source": source_url})

    if not out_faqs:
        # Totale fallback – heel beknopt, zonder bron
        _log("INFO", "faq_generated", context="none", n=0)
        stub = [{"q": f"What is {topic} ({i+1})?", "a": f"{topic.capitalize()} explained (stub).", "source": None}
                for i in range(count)]
        return {"_context_used": "none", "faqs": stub, "topic": topic}

    _log("INFO", "faq_generated", context=ctx_used, n=len(out_faqs))
    return {"_context_used": ctx_used, "faqs": out_faqs, "topic": topic}
