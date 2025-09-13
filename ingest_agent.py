# ingest_agent.py
import os, json, time, hashlib
from typing import List, Dict, Any, Optional
from openai import OpenAI
from openai._exceptions import OpenAIError, APIConnectionError, RateLimitError

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "20"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))

INGEST_TIME_BUDGET_SEC   = int(os.getenv("INGEST_TIME_BUDGET_SEC", "90"))
INGEST_MAX_CHUNKS_TOTAL  = int(os.getenv("INGEST_MAX_CHUNKS_TOTAL", "120"))
INGEST_MAX_CHUNKS_PAGE   = int(os.getenv("INGEST_MAX_CHUNKS_PER_PAGE", "8"))

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def _chunk_text(text: str, max_chars: int = 1500, overlap: int = 150) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks, i, n = [], 0, len(text)
    while i < n and len(chunks) < INGEST_MAX_CHUNKS_PAGE:
        end = min(i + max_chars, n)
        piece = text[i:end]
        if piece.strip():
            chunks.append(piece)
        i = end - overlap
        if i < 0: i = 0
        if i >= n: break
    return chunks

def _page_snippet(p: Dict[str, Any]) -> str:
    bits: List[str] = []
    for k in ("title","h1","meta_description"):
        v = p.get(k)
        if v: bits.append(str(v))
    for h in (p.get("h2") or [])[:5]: bits.append(h)
    for h in (p.get("h3") or [])[:5]: bits.append(h)
    for para in (p.get("paragraphs") or [])[:2]:
        if isinstance(para, str) and para.strip():
            bits.append(para.strip())
    return "\n".join([b for b in bits if b]).strip()

def _embed_with_retry(text: str) -> Optional[List[float]]:
    last_err = None
    for attempt in range(1, OPENAI_MAX_RETRIES + 1):
        try:
            resp = client.embeddings.create(
                model=EMBED_MODEL,
                input=[text],
                timeout=OPENAI_TIMEOUT_SEC,
            )
            return resp.data[0].embedding
        except (RateLimitError, APIConnectionError, OpenAIError, TimeoutError, Exception) as e:
            last_err = e
            backoff = 1.5 ** (attempt - 1)
            print(json.dumps({"level":"WARN","msg":"embedding_retry","attempt":attempt,"error":str(e)[:200]}), flush=True)
            time.sleep(backoff)
    print(json.dumps({"level":"ERROR","msg":"embedding_failed","error":str(last_err)[:300]}), flush=True)
    return None

def _hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

def ingest_crawl_output(conn, site_id: str, crawl_output: Dict[str, Any]) -> int:
    started = time.time()
    pages = (crawl_output or {}).get("pages") or []
    if not pages:
        return 0

    inserted = 0
    with conn.cursor() as cur:
        for p_idx, p in enumerate(pages):
            if time.time() - started > INGEST_TIME_BUDGET_SEC or inserted >= INGEST_MAX_CHUNKS_TOTAL:
                print(json.dumps({"level":"WARN","msg":"ingest_budget_reached","after_pages":p_idx,"inserted":inserted}), flush=True)
                break

            url = p.get("final_url") or p.get("url")
            if not url:
                continue

            content = _page_snippet(p)
            if not content:
                continue

            for chunk in _chunk_text(content):
                if time.time() - started > INGEST_TIME_BUDGET_SEC or inserted >= INGEST_MAX_CHUNKS_TOTAL:
                    break

                chash = _hash(chunk)

                cur.execute("""
                    SELECT 1 FROM documents
                     WHERE site_id=%s AND url=%s AND content_hash=%s
                     LIMIT 1
                """, (site_id, url, chash))
                if cur.fetchone():
                    continue

                emb = _embed_with_retry(chunk)
                if emb is None:
                    continue

                try:
                    cur.execute("""
                        INSERT INTO documents (site_id, url, language, content, metadata, embedding, content_hash)
                        VALUES (%s, %s, NULL, %s, %s, %s, %s)
                    """, (site_id, url, chunk, json.dumps({"source":"crawl"}), emb, chash))
                    inserted += 1
                except Exception as db_err:
                    try: conn.rollback()
                    except Exception: pass
                    print(json.dumps({"level":"ERROR","msg":"ingest_insert_failed","url":url,"error":str(db_err)[:300]}), flush=True)
                    cur = conn.cursor()
                    continue

            try:
                conn.commit()
            except Exception as c_err:
                try: conn.rollback()
                except Exception: pass
                print(json.dumps({"level":"ERROR","msg":"ingest_commit_failed","error":str(c_err)[:300]}), flush=True)

    print(json.dumps({"level":"INFO","msg":"ingest_done","chunks":inserted,"duration_ms": int((time.time() - started) * 1000)}), flush=True)
    return inserted
