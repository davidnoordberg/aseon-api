# ingest_agent.py
# Ingests crawl output -> pgvector documents table
# - memory-slim: stream per page, small batches
# - robust: explicit ::vector cast, rollbacks on error
# - configurable via env

from __future__ import annotations
import os, json, math, time
from typing import List, Dict, Any, Iterable, Tuple

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json
from openai import OpenAI

# -------------------- Config --------------------

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
# Chunking
MAX_CHARS = int(os.getenv("INGEST_MAX_CHARS", "1800"))
OVERLAP   = int(os.getenv("INGEST_OVERLAP", "200"))
# Batching embeddings -> minder API overhead / mem pressure
BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "8"))
# Optioneel throttling om spikes te dempen (0 = uit)
SLEEP_BETWEEN_BATCHES = float(os.getenv("INGEST_SLEEP_BETWEEN_BATCHES", "0"))
# Skip lege/te korte chunks
MIN_CHARS = int(os.getenv("INGEST_MIN_CHARS", "40"))

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# -------------------- Helpers --------------------

def _chunk_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> Iterable[str]:
    """Plain char-based chunking (fast/robust)."""
    if not text:
        return []
    n = len(text)
    i = 0
    out: List[str] = []
    while i < n:
        end = min(i + max_chars, n)
        chunk = text[i:end].strip()
        if len(chunk) >= MIN_CHARS:
            out.append(chunk)
        nxt = end - overlap
        if nxt <= i:  # guard
            nxt = end
        i = nxt
    return out

def _page_to_textbits(p: Dict[str, Any]) -> List[str]:
    bits: List[str] = []
    # title / h1 / meta
    for k in ("title", "h1", "meta_description"):
        v = p.get(k)
        if v: bits.append(str(v))
    # h2 / h3 (lists)
    for h in (p.get("h2") or []):
        if h: bits.append(str(h))
    for h in (p.get("h3") or []):
        if h: bits.append(str(h))
    # Optioneel: korte “paragraphs” die je crawler kan zetten
    for para in (p.get("paragraphs") or [])[:3]:
        if para and isinstance(para, str):
            bits.append(para)
    return bits

def _build_content(p: Dict[str, Any]) -> str:
    return "\n".join(_page_to_textbits(p)).strip()

def _batched(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

# -------------------- Public API --------------------

def ingest_crawl_output(conn, site_id: str, crawl_output: Dict[str, Any]) -> int:
    """
    Neemt crawl_output (zoals uit crawl_light) en schrijft chunks + embeddings
    naar 'documents' (site_id, url, language, content, metadata, embedding).
    Returnt het aantal ingevoegde chunks.
    """
    pages = (crawl_output or {}).get("pages") or []
    if not pages:
        return 0

    inserted_total = 0

    try:
        with conn.cursor() as cur:
            for p in pages:
                url = p.get("final_url") or p.get("url")
                if not url:
                    continue

                # taal (optioneel) — crawler kan language meegeven; anders NULL
                language = p.get("language")

                content = _build_content(p)
                if not content:
                    continue

                # Build chunks
                chunks = list(_chunk_text(content))
                if not chunks:
                    continue

                # Embed in batches
                for batch in _batched(chunks, BATCH_SIZE):
                    # 1) Embeddings call (batched)
                    resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
                    vecs = [d.embedding for d in resp.data]  # list[list[float]]

                    # 2) Insert rows (explicit ::vector cast)
                    params: List[Tuple[Any, ...]] = []
                    for chunk, vec in zip(batch, vecs):
                        meta = {"source": "crawl"}
                        params.append((site_id, url, language, chunk, Json(meta), vec))

                    # single executemany to minimize roundtrips
                    cur.executemany(
                        """
                        INSERT INTO documents (site_id, url, language, content, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s::vector)
                        """,
                        params,
                    )
                    inserted_total += len(batch)

                    # Flush per batch to keep memory low and avoid long xacts
                    conn.commit()

                    if SLEEP_BETWEEN_BATCHES > 0:
                        time.sleep(SLEEP_BETWEEN_BATCHES)

        # Final ensure
        conn.commit()
        return inserted_total

    except Exception as e:
        # Zorg dat de transactie niet 'aborted' blijft
        try:
            conn.rollback()
        except Exception:
            pass
        # Propagate verder; general_agent logt als ingest_failed
        raise

# -------------------- Quick CLI smoke (optional) --------------------
if __name__ == "__main__":
    # Minimal smoketest (expects DATABASE_URL env & a tiny crawl_output JSON on stdin)
    import sys
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)
    payload = json.load(sys.stdin)
    with psycopg.connect(dsn, row_factory=dict_row) as c:
        site_id = os.environ.get("SITE_ID") or payload.get("site_id") or "00000000-0000-0000-0000-000000000000"
        n = ingest_crawl_output(c, site_id, payload)
        print(json.dumps({"inserted": n}))
