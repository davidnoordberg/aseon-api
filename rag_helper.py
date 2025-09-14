# rag_helper.py
import os, json, re, time
from typing import List, Dict, Any, Optional
from psycopg.rows import dict_row
from openai import OpenAI

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "20"))
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def _trim(s: str, max_chars: int = 1200) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s[:max_chars]

def embed(text: str) -> List[float]:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text or ""],
        timeout=OPENAI_TIMEOUT_SEC,
    )
    return resp.data[0].embedding

def search_site_docs(conn, site_id: str, query: str, k: int = 8) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        qvec = embed(query)
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT url, content, metadata
                  FROM documents
                 WHERE site_id=%s AND embedding IS NOT NULL
                 ORDER BY embedding <-> %s::vector
                 LIMIT %s
            """, (site_id, qvec, k))
            rows = cur.fetchall()
    except Exception as e:
        try: conn.rollback()
        except Exception: pass
        print(json.dumps({"level":"WARN","msg":"site_vec_failed","error":str(e)[:200]}), flush=True)

    if not rows:
        try:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("""
                    SELECT url, content, metadata
                      FROM documents
                     WHERE site_id=%s AND (content ILIKE %s OR url ILIKE %s)
                     LIMIT %s
                """, (site_id, f"%{query}%", f"%{query}%", k))
                rows = cur.fetchall()
        except Exception as e:
            try: conn.rollback()
            except Exception: pass
            print(json.dumps({"level":"ERROR","msg":"site_like_failed","error":str(e)[:200]}), flush=True)
    return rows or []

def search_kb(conn, query: str, k: int = 6, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    where = ""
    params: List[Any] = []
    if tags:
        where = "WHERE tags && %s"
        params.append(tags)
    try:
        qvec = embed(query)
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(f"""
                SELECT title, url, source, tags, content
                  FROM kb_documents
                  {where}
                 ORDER BY embedding <-> %s::vector
                 LIMIT %s
            """, (*params, qvec, k))
            rows = cur.fetchall()
    except Exception as e:
        try: conn.rollback()
        except Exception: pass
        print(json.dumps({"level":"WARN","msg":"kb_vec_failed","error":str(e)[:200]}), flush=True)

    if not rows:
        try:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(f"""
                    SELECT title, url, source, tags, content
                      FROM kb_documents
                      {where + (' AND ' if where else ' WHERE ')} (content ILIKE %s OR title ILIKE %s)
                     LIMIT %s
                """, (*params, f"%{query}%", f"%{query}%", k))
                rows = cur.fetchall()
        except Exception as e:
            try: conn.rollback()
            except Exception: pass
            print(json.dumps({"level":"ERROR","msg":"kb_like_failed","error":str(e)[:200]}), flush=True)
    return rows or []

def build_context(site_rows, kb_rows) -> Dict[str, Any]:
    site_bits, kb_bits = [], []
    site_cites, kb_cites = [], []

    for i, r in enumerate(site_rows[:8]):
        sid = f"S{i+1}"
        site_cites.append({"id": sid, "url": r.get("url")})
        site_bits.append(f"[{sid}] {r.get('url')}\n{_trim(r.get('content') or '')}")

    for i, r in enumerate(kb_rows[:6]):
        kid = f"K{i+1}"
        title = r.get("title") or (r.get("source") or "KB")
        kb_cites.append({"id": kid, "url": r.get("url"), "title": title})
        kb_bits.append(f"[{kid}] {title} — {r.get('url')}\n{_trim(r.get('content') or '')}")

    return {
        "site_ctx": "\n\n".join(site_bits),
        "kb_ctx": "\n\n".join(kb_bits),
        "site_citations": site_cites,
        "kb_citations": kb_cites
    }

def get_rag_context(conn, site_id: str, query: str,
                    k_site: int = 8, k_kb: int = 6,
                    kb_tags: Optional[List[str]] = None) -> Dict[str, Any]:
    srows = search_site_docs(conn, site_id, query, k=k_site)
    krows = search_kb(conn, query, k=k_kb, tags=kb_tags)
    return build_context(srows, krows)
