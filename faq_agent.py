# faq_agent.py
import os, json, re
from typing import Dict, Any, List
import psycopg
from psycopg.rows import dict_row
from openai import OpenAI

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def _embed(text: str) -> List[float]:
    return client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding

def _make_context_from_documents(conn, site_id: str, topic: str, k: int = 5) -> tuple[str, str]:
    from psycopg.rows import dict_row
    try:
        qvec = _embed(topic)
        with conn.cursor(row_factory=dict_row) as cur:
            # CAST het parameter-argument expliciet naar vector
            cur.execute("""
                SELECT url, content
                  FROM documents
                 WHERE site_id = %s
                 ORDER BY embedding <-> %s::vector
                 LIMIT %s
            """, (site_id, qvec, k))
            rows = cur.fetchall()
        if rows:
            ctx = "\n\n".join([f"[{i+1}] {r['url']}\n{r['content']}" for i, r in enumerate(rows)])
            return ctx, "documents"
    except Exception as e:
        # heel belangrijk: abort terugdraaien, anders faalt de fallback ook
        try: conn.rollback()
        except Exception: pass
        print(json.dumps({"level":"WARN","msg":"faq_vector_search_failed","error":str(e)}), flush=True)

    # LIKE-fallback in een schone transactie
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT url, content
                  FROM documents
                 WHERE site_id=%s AND (content ILIKE %s OR content ILIKE %s)
                 LIMIT %s
            """, (site_id, f"%{topic}%", "%SEO%", k))
            rows = cur.fetchall()
        if rows:
            ctx = "\n\n".join([f"[{i+1}] {r['url']}\n{r['content']}" for i, r in enumerate(rows)])
            return ctx, "like"
    except Exception as e:
        try: conn.rollback()
        except Exception: pass
        print(json.dumps({"level":"ERROR","msg":"faq_like_search_failed","error":str(e)}), flush=True)

    return "", "none"


def generate_faqs(conn, site_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    topic = (payload or {}).get("topic") or "general"
    count = int((payload or {}).get("count", 6))
    max_words = int((payload or {}).get("max_words", 80))
    use_context = (payload or {}).get("use_context", "auto")

    ctx_text, ctx_label = ("", "none")
    if use_context in ("auto","documents","crawl"):
        ctx_text, ctx_label = _make_context_from_documents(conn, site_id, topic, k=5)

    sys = f"""You write concise, factual FAQs grounded in the given context.
Rules:
- {count} QA pairs, each answer ≤ {max_words} words.
- If the context includes URLs in square brackets like [1] https://..., prefer citing one of those URLs as 'source'.
- No marketing fluff; be specific.
Return JSON: {{ "faqs": [{{"q": "...","a":"...","source":"<url or null>"}}...] }}"""

    user = f"Topic: {topic}\n\nContext:\n{ctx_text or '(no context)'}"
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        response_format={"type":"json_object"},
        temperature=0.2,
    )

    try:
        data = json.loads(resp.choices[0].message.content)
        faqs = data.get("faqs") or []
    except Exception:
        faqs = []

    # sanitize + enforce word limit
    out = []
    for f in faqs[:count]:
        q = (f.get("q") or "").strip()
        a = re.sub(r"\s+", " ", (f.get("a") or "").strip())
        if not q or not a: continue
        words = a.split(" ")
        if len(words) > max_words:
            a = " ".join(words[:max_words])
        src = f.get("source") or None
        out.append({"q": q, "a": a, "source": src})

    if not out:
        # deterministic fallback (avoid empty output)
        out = [{"q": f"What is {topic} ({i+1})?", "a": f"{topic.capitalize()} explained (stub).", "source": None} for i in range(count)]

    print(json.dumps({"level":"INFO","msg":"faq_generated","context":ctx_label,"n":len(out)}), flush=True)
    return {"_context_used": (ctx_label if out and ctx_label!="none" else None), "faqs": out}
