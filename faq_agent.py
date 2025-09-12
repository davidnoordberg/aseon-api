# faq_agent.py
import os, json, math, re
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone
import psycopg
from psycopg.rows import dict_row
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
MAX_WORDS_DEFAULT = int(os.getenv("FAQ_MAX_WORDS", "80"))
TOP_K = int(os.getenv("FAQ_TOP_K", "5"))

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def _log(level: str, msg: str, **kwargs):
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level.upper(),
        "msg": msg,
        **kwargs
    }
    print(json.dumps(payload), flush=True)

_word_re = re.compile(r"\S+")

def _trim_words(s: str, max_words: int) -> str:
    if not s: return ""
    words = _word_re.findall(s)
    if len(words) <= max_words:
        return s.strip()
    return " ".join(words[:max_words]).strip()

def _fetch_documents(conn, site_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT url, content, metadata
            FROM documents
            WHERE site_id = %s
            ORDER BY url
            LIMIT %s
        """, (site_id, limit))
        return list(cur.fetchall())

def _search_documents(conn, site_id: str, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    # Vector search als embeddings aanwezig zijn; zo niet, fallback op LIKE
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT url, content, metadata
                FROM documents
                WHERE site_id = %s
                LIMIT 1
            """, (site_id,))
            has_docs = cur.fetchone() is not None
    except Exception:
        has_docs = False

    if not has_docs:
        return []

    # probeer vector-zoek
    try:
        emb = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT url, content, metadata
                FROM documents
                WHERE site_id = %s
                ORDER BY embedding <-> %s
                LIMIT %s
            """, (site_id, emb, top_k))
            rows = list(cur.fetchall())
            return rows
    except Exception as e:
        _log("warn", "faq_vector_search_failed", error=str(e))

    # LIKE fallback
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT url, content, metadata
                FROM documents
                WHERE site_id = %s AND content ILIKE %s
                ORDER BY url
                LIMIT %s
            """, (site_id, f"%{query}%", top_k))
            return list(cur.fetchall())
    except Exception as e:
        _log("error", "faq_like_search_failed", error=str(e))
        return []

def _crawl_context(conn, site_id: str, top_n: int = TOP_K) -> List[Dict[str, Any]]:
    # Haal laatste crawl-output op en maak kleine contextsnippets
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT output
            FROM jobs
            WHERE site_id = %s AND type = 'crawl' AND status = 'done'
            ORDER BY finished_at DESC NULLS LAST, created_at DESC
            LIMIT 1
        """, (site_id,))
        row = cur.fetchone()
    if not row or not row.get("output"):
        return []

    out = row["output"] or {}
    pages = out.get("pages") or []
    snippets = []
    for p in pages:
        url = p.get("final_url") or p.get("url")
        if not url: 
            continue
        parts = []
        for k in ("title", "h1"):
            v = p.get(k)
            if v: parts.append(str(v))
        for h in (p.get("h2") or [])[:4]: parts.append(h)
        for h in (p.get("h3") or [])[:4]: parts.append(h)
        meta = p.get("meta_description")
        if meta: parts.append(meta)
        if not parts: 
            continue
        content = " • ".join(parts)
        snippets.append({"url": url, "content": content})
        if len(snippets) >= top_n:
            break
    return snippets

def _build_prompt(topic: str,
                  max_words: int,
                  seed_qs: Optional[List[str]],
                  contexts: List[Dict[str, Any]],
                  language: str = "en") -> str:
    # Contextblok met bronnen
    ctx_lines = []
    for c in contexts:
        url = c.get("url", "")
        txt = c.get("content", "")[:600]
        ctx_lines.append(f"- Source: {url}\n  Excerpt: {txt}")
    ctx_block = "\n".join(ctx_lines) if ctx_lines else "(no context)"

    # optionele seed questions
    sq_block = ""
    if seed_qs:
        sq_block = "SeedQuestions:\n" + "\n".join([f"- {q}" for q in seed_qs]) + "\n"

    return f"""You are an SEO/AEO assistant writing FAQ for featured snippets in {language}.

Topic: "{topic}"

Context (site excerpts):
{ctx_block}

Requirements:
- 5–10 Q&A pairs (we'll pass 'count').
- Answers MUST be <= {max_words} words. Keep them factual and concise.
- Use only facts implied by the context; do not invent data like addresses, prices, phones.
- Each answer should be standalone and snippet-ready.
- Return JSON with field "faqs": [{{"q": "...", "a": "...", "source": "<url or null>"}}].

{sq_block}
Return ONLY valid JSON (no commentary).
"""

def _llm_build_faq(topic: str,
                   count: int,
                   max_words: int,
                   contexts: List[Dict[str, Any]],
                   language: str = "en",
                   seed_qs: Optional[List[str]] = None) -> Dict[str, Any]:
    prompt = _build_prompt(topic, max_words, seed_qs, contexts, language=language)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        data = {"faqs": []}

    faqs = data.get("faqs") or []
    # Normaliseer velden en enforce max_words
    norm = []
    for item in faqs[:count]:
        q = (item.get("q") or item.get("question") or "").strip()
        a = (item.get("a") or item.get("answer") or "").strip()
        src = item.get("source")
        a = _trim_words(a, max_words)
        norm.append({"q": q, "a": a, "source": src if isinstance(src, str) and src else None})
    return {"faqs": norm}

def generate_faqs(conn,
                  site_id: str,
                  payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Payload:
      topic: str
      count: int (5..10)
      use_context: "auto" | "documents" | "crawl" | "none"  (default "auto")
      max_words: int (default MAX_WORDS_DEFAULT)
      language: str (default "en")
      seed_questions: [str] (optional)
    """
    topic = (payload or {}).get("topic") or "general"
    count = int((payload or {}).get("count", 6))
    count = max(5, min(10, count))
    use_context = (payload or {}).get("use_context") or "auto"
    max_words = int((payload or {}).get("max_words", MAX_WORDS_DEFAULT))
    language = (payload or {}).get("language", "en")
    seed_qs = (payload or {}).get("seed_questions") or None

    contexts: List[Dict[str, Any]] = []
    context_used = "none"

    try:
        if use_context in ("auto", "documents"):
            # Probeer document RAG
            # 1) heeft site documenten?
            docs = _fetch_documents(conn, site_id, limit=200)
            if docs:
                # Query afleiden uit topic
                q = f"{topic} FAQ for this site"
                top = _search_documents(conn, site_id, q, top_k=TOP_K)
                if top:
                    contexts = [{"url": r["url"], "content": r["content"]} for r in top]
                    context_used = "documents"
        if not contexts and use_context in ("auto", "crawl"):
            # Fallback naar laatste crawl(snippets)
            crawl_ctx = _crawl_context(conn, site_id, top_n=TOP_K)
            if crawl_ctx:
                contexts = crawl_ctx
                context_used = "crawl"
        if use_context == "none":
            contexts = []
            context_used = "none"
    except Exception as e:
        _log("warn", "faq_context_build_failed", error=str(e))

    out = _llm_build_faq(topic, count, max_words, contexts, language=language, seed_qs=seed_qs)

    # Als er context is, maar LLM geen bronnen zette → heuristisch bron invullen (eigen domein eerst)
    if contexts:
        base_url = None
        # kies eerste context-url als fallback bron
        for c in contexts:
            u = c.get("url")
            if u:
                base_url = u
                break
        fixed = []
        for f in out.get("faqs", []):
            src = f.get("source")
            if not src:
                f["source"] = base_url
            fixed.append(f)
        out["faqs"] = fixed

    out["_context_used"] = context_used
    _log("info", "faq_generated", context=context_used, n=len(out.get("faqs", [])))
    return out
