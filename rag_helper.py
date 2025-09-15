# rag_helper.py
import os, json, re, time, math
from typing import List, Dict, Any, Optional, Tuple
from psycopg.rows import dict_row
from openai import OpenAI
from random import random

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # 1536-dim
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "20"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
CONTEXT_CHAR_BUDGET = int(os.getenv("RAG_CHAR_BUDGET", "9000"))  # totale contextlimiet

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def _trim(s: str, max_chars: int = 1200) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s[:max_chars]

def _retry_sleep(attempt: int) -> float:
    # jittered backoff
    return min(2 ** attempt + random(), 8.0)

def embed(text: str) -> List[float]:
    """1-shot embed met retries; geeft zero-vector terug als alles faalt."""
    last_err = None
    for attempt in range(OPENAI_MAX_RETRIES):
        try:
            resp = client.embeddings.create(
                model=EMBED_MODEL,
                input=[text or ""],
                timeout=OPENAI_TIMEOUT_SEC,
            )
            return resp.data[0].embedding
        except Exception as e:
            last_err = e
            time.sleep(_retry_sleep(attempt))
    # fallback: zero-vector met correcte dimensie (1536 voor text-embedding-3-small)
    dim = 1536 if "small" in EMBED_MODEL else (3072 if "large" in EMBED_MODEL else 1536)
    print(json.dumps({"level":"ERROR","msg":"embed_failed","error":str(last_err)[:300]}), flush=True)
    return [0.0]*dim

def _parse_tags(tags: Optional[List[str]]) -> Optional[List[str]]:
    if not tags: return None
    return [t.strip() for t in tags if t and t.strip()]

def search_site_docs(conn, site_id: str, query: str, k: int = 8) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        qvec = embed(query)
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT url, content, metadata, (embedding <-> %s::vector) AS dist
                  FROM documents
                 WHERE site_id=%s AND embedding IS NOT NULL
              ORDER BY dist ASC
                 LIMIT %s
            """, (qvec, site_id, k))
            rows = cur.fetchall()
    except Exception as e:
        try: conn.rollback()
        except Exception: pass
        print(json.dumps({"level":"WARN","msg":"site_vec_failed","error":str(e)[:200]}), flush=True)

    if not rows:
        # tekstuele fallback
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
    tags = _parse_tags(tags)
    where = ""
    params: List[Any] = []
    if tags:
        where = "WHERE tags && %s"
        params.append(tags)
    try:
        qvec = embed(query)
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(f"""
                SELECT title, url, source, tags, content, (embedding <-> %s::vector) AS dist
                  FROM kb_documents
                  {where if where else ""}
              ORDER BY dist ASC
                 LIMIT %s
            """, (qvec, *params, k) if tags else (qvec, k))
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
                """, (*params, f"%{query}%", f"%{query}%", k) if tags else (f"%{query}%", f"%{query}%", k))
                rows = cur.fetchall()
        except Exception as e:
            try: conn.rollback()
            except Exception: pass
            print(json.dumps({"level":"ERROR","msg":"kb_like_failed","error":str(e)[:200]}), flush=True)
    return rows or []

def _collapse_by_url(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in rows:
        u = (r.get("url") or "").strip()
        if u and u not in seen:
            seen.add(u)
            out.append(r)
    return out

def build_context(site_rows, kb_rows, budget_chars: int = CONTEXT_CHAR_BUDGET) -> Dict[str, Any]:
    site_rows = _collapse_by_url(site_rows)[:8]
    kb_rows = _collapse_by_url(kb_rows)[:6]

    site_bits, kb_bits = [], []
    site_cites, kb_cites = [], []
    used = 0

    # eerst site, dan kb — binnen totaalbudget
    for i, r in enumerate(site_rows):
        sid = f"S{i+1}"
        chunk = f"[{sid}] {r.get('url')}\n{_trim(r.get('content') or '')}\n"
        if used + len(chunk) > budget_chars: break
        used += len(chunk)
        site_cites.append({"id": sid, "url": r.get("url")})
        site_bits.append(chunk)

    for i, r in enumerate(kb_rows):
        kid = f"K{i+1}"
        title = r.get("title") or (r.get("source") or "KB")
        chunk = f"[{kid}] {title} — {r.get('url')}\n{_trim(r.get('content') or '')}\n"
        if used + len(chunk) > budget_chars: break
        used += len(chunk)
        kb_cites.append({"id": kid, "url": r.get("url"), "title": title})
        kb_bits.append(chunk)

    return {
        "site_ctx": "\n".join(site_bits).strip(),
        "kb_ctx": "\n".join(kb_bits).strip(),
        "site_citations": site_cites,
        "kb_citations": kb_cites,
        "char_used": used,
        "char_budget": budget_chars
    }

def get_rag_context(conn, site_id: str, query: str,
                    k_site: int = 8, k_kb: int = 6,
                    kb_tags: Optional[List[str]] = None) -> Dict[str, Any]:
    srows = search_site_docs(conn, site_id, query, k=k_site)
    krows = search_kb(conn, query, k=k_kb, tags=kb_tags)
    return build_context(srows, krows)
