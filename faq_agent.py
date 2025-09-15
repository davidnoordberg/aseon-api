import os, json, re, time
from typing import Dict, Any, List, Tuple
import psycopg
from psycopg.rows import dict_row
from openai import OpenAI

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "20"))

# Context limieten
MAX_CTX_CHARS = int(os.getenv("FAQ_MAX_CTX_CHARS", "3500"))
DOC_K = int(os.getenv("FAQ_DOC_TOPK", "5"))
KB_K  = int(os.getenv("FAQ_KB_TOPK", "4"))

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def _embed(text: str) -> List[float]:
    text = (text or "").strip()
    return client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding

def _rows_to_ctx(rows: List[dict], label: str) -> str:
    bits: List[str] = []
    for i, r in enumerate(rows):
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip()
        content = (r.get("content") or "").strip()
        pre = f"[{label.upper()} {i+1}] {url}"
        if title: pre += f" • {title}"
        snippet = (content[:1000] + "…") if len(content) > 1000 else content
        if snippet:
            bits.append(pre + "\n" + snippet)
    return "\n\n".join(bits)

def _query_site_documents(conn, site_id: str, topic: str, k: int) -> List[dict]:
    """
    Let op: 'documents' tabel heeft geen 'title' kolom. We halen waar mogelijk de titel
    uit metadata->>'title' en bouwen een snippet op basis van 'content'.
    """
    try:
        qvec = _embed(topic)
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT
                    url,
                    COALESCE(metadata->>'title','') AS title,
                    content
                FROM documents
                WHERE site_id = %s
                ORDER BY embedding <-> %s::vector
                LIMIT %s
            """, (site_id, qvec, k))
            return cur.fetchall() or []
    except Exception as e:
        try: conn.rollback()
        except Exception: pass
        print(json.dumps({"level":"WARN","msg":"faq_site_vector_failed","error":str(e)[:300]}), flush=True)
        return []

def _query_kb(conn, topic: str, k: int) -> List[dict]:
    try:
        qvec = _embed(topic)
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT url, title, content
                FROM kb_documents
                ORDER BY embedding <-> %s::vector
                LIMIT %s
            """, (qvec, k))
            return cur.fetchall() or []
    except Exception as e:
        try: conn.rollback()
        except Exception: pass
        print(json.dumps({"level":"WARN","msg":"faq_kb_vector_failed","error":str(e)[:300]}), flush=True)
        return []

def _fallback_crawl_snapshot(conn, site_id: str, max_pages: int = 6) -> str:
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT output
                FROM jobs
                WHERE site_id=%s AND type='crawl' AND status='done'
                ORDER BY COALESCE(finished_at, created_at) DESC
                LIMIT 1
            """, (site_id,))
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
        try: conn.rollback()
        except Exception: pass
        print(json.dumps({"level":"WARN","msg":"faq_crawl_ctx_failed","error":str(e)[:300]}), flush=True)
        return ""

def _trim_ctx(*chunks: Tuple[str, str]) -> Tuple[str, List[str]]:
    used: List[str] = []
    buf: List[str] = []
    total = 0
    labels_map = {"SITE":"documents","KB":"kb","CRAWL":"crawl"}
    for label, text in chunks:
        if not text: continue
        t = text.strip()
        if not t: continue
        add = len(t)
        if total + add > MAX_CTX_CHARS:
            t = t[: max(0, MAX_CTX_CHARS - total)]
            add = len(t)
        if add <= 0: break
        buf.append(t)
        total += add
        if label in labels_map and labels_map[label] not in used:
            used.append(labels_map[label])
        if total >= MAX_CTX_CHARS: break
    return ("\n\n".join(buf), used)

def generate_faqs(conn, site_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    topic = (payload or {}).get("topic") or "general"
    count = int((payload or {}).get("count", 6))
    max_words = int((payload or {}).get("max_words", 80))
    use_context = (payload or {}).get("use_context", "auto")

    site_rows: List[dict] = []
    kb_rows: List[dict] = []
    ctx_used: List[str] = []

    if use_context in ("auto", "documents", "kb", "crawl"):
        site_rows = _query_site_documents(conn, site_id, topic, DOC_K)
        site_ctx = _rows_to_ctx(site_rows, label="site")
        kb_rows = _query_kb(conn, topic, KB_K)
        kb_ctx = _rows_to_ctx(kb_rows, label="kb")
        crawl_ctx = _fallback_crawl_snapshot(conn, site_id, max_pages=6)
        ctx_text, ctx_used = _trim_ctx(
            ("SITE", site_ctx),
            ("KB",   kb_ctx),
            ("CRAWL", crawl_ctx),
        )
    else:
        ctx_text, ctx_used = "(no context)", []

    sys = f"""You write concise, factual FAQs grounded ONLY in the provided context.
Return EXACTLY {count} Q/A pairs. Each answer ≤ {max_words} words, cite one best source URL when available.
If info is missing, say you cannot answer from context. JSON ONLY:
{{ "faqs": [{{"q":"...","a":"...","source":"<url|null>"}}] }}"""
    user = f"Topic: {topic}\n\nContext:\n{ctx_text or '(no context)'}"

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        response_format={"type":"json_object"},
        temperature=0.2,
        timeout=OPENAI_TIMEOUT_SEC,
    )

    try:
        data = json.loads(resp.choices[0].message.content)
        faqs = data.get("faqs") or []
    except Exception:
        faqs = []

    out_faqs = []
    for f in faqs[:count]:
        q = (f.get("q") or "").strip()
        a = re.sub(r"\s+", " ", (f.get("a") or "").strip())
        if not q or not a: continue
        words = a.split(" ")
        if len(words) > max_words:
            a = " ".join(words[:max_words])
        src = (f.get("source") or None)
        out_faqs.append({"q": q, "a": a, "source": src})

    if not out_faqs:
        out_faqs = [{"q": f"What is {topic} ({i+1})?", "a": f"{topic.capitalize()} explained (stub).", "source": None} for i in range(count)]

    print(json.dumps({"level":"INFO","msg":"faq_generated","context_used":ctx_used,"n":len(out_faqs)}), flush=True)
    return {"_context_used": (ctx_used or None), "faqs": out_faqs}
