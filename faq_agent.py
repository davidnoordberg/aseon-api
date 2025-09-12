# faq_agent.py
import os, json, math
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import psycopg
from psycopg.rows import dict_row

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

_FAQ_SYS = """
You are an AEO (Answer Engine Optimization) assistant.
Produce 5–10 concise, factual Q&A pairs suitable for featured snippets and AI answers.
Rules:
- ≤80 words per answer.
- Cite source URLs from the site for each answer.
- No marketing fluff. Be clear, specific, and action-oriented.
- Use the provided context only; do not fabricate facts.
Return JSON: { "faqs": [ { "q": "...", "a": "...", "source": "https://..." } ] }
"""

def _llm_faq(topic: str, context: str, n: int = 6) -> Dict[str, Any]:
    prompt = f"""
Topic: {topic}
How many: {n}
Context (site pages & content excerpts):
{context}
"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": _FAQ_SYS}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        data = {"faqs": []}
    # enforce shape & limits
    out = []
    for item in (data.get("faqs") or [])[:10]:
        q = (item.get("q") or "").strip()
        a = (item.get("a") or "").strip()
        src = (item.get("source") or "").strip()
        if q and a and src:
            out.append({"q": q, "a": a[:1200], "source": src})
    return {"faqs": out}

def _fetch_latest_crawl(conn, site_id: str) -> Optional[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT output FROM jobs
             WHERE site_id=%s AND type='crawl' AND status='done'
         ORDER BY COALESCE(finished_at, created_at) DESC
            LIMIT 1
        """, (site_id,))
        row = cur.fetchone()
        return row["output"] if row else None

def _crawl_to_context(crawl: Dict[str, Any], k_pages: int = 12, max_chars: int = 14000) -> str:
    pages = (crawl or {}).get("pages") or []
    sel = []
    for p in pages[:k_pages]:
        sel.append({
            "url": p.get("url"),
            "title": p.get("title"),
            "h1": p.get("h1"),
            "meta_description": p.get("meta_description")
        })
    text = json.dumps({"pages": sel}, ensure_ascii=False)
    return text[:max_chars]

def _pgvector_available(conn) -> bool:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.documents') IS NOT NULL AS ok;")
            return bool(cur.fetchone()["ok"])
    except Exception:
        return False

def _semantic_context_from_documents(conn, site_id: str, topic: str, k: int = 5) -> List[Dict[str, Any]]:
    # Simple semantic search using cosine distance if available; otherwise empty
    try:
        # embed topic
        emb = client.embeddings.create(model=EMBED_MODEL, input=[topic]).data[0].embedding
        vec = "[" + ",".join(f"{x:.8f}" for x in emb) + "]"
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT url, content, 1 - (embedding <=> '{vec}'::vector) AS score
                  FROM documents
                 WHERE site_id=%s
              ORDER BY embedding <=> '{vec}'::vector
                 LIMIT %s
            """, (site_id, k))
            return cur.fetchall()
    except Exception:
        return []

def _build_context(conn, site_id: str, topic: str) -> Tuple[str, str]:
    """
    Returns (context_text, context_kind) where kind in {'documents','crawl','none'}
    """
    if _pgvector_available(conn):
        docs = _semantic_context_from_documents(conn, site_id, topic, k=6)
        if docs:
            bundle = [{"url": d["url"], "excerpt": (d["content"] or "")[:600]} for d in docs]
            return json.dumps({"docs": bundle}, ensure_ascii=False), "documents"

    crawl = _fetch_latest_crawl(conn, site_id)
    if crawl:
        return _crawl_to_context(crawl), "crawl"

    return "(no site context available)", "none"

def generate_faqs(conn, site_id: str, topic: Optional[str] = None, n: int = 6) -> Dict[str, Any]:
    t = (topic or "general").strip()
    ctx_text, kind = _build_context(conn, site_id, t)
    data = _llm_faq(topic=t, context=ctx_text, n=max(5, min(10, n)))
    data["_context_used"] = kind
    data["topic"] = t
    return data
