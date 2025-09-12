# ingest_agent.py
# V14: streaming & low-memory ingest
# - Chunk per page, hard caps
# - Microbatch embeddings with retries
# - Immediate INSERT after each batch (no large in-RAM buffers)
# - Aggressive GC to keep RSS stable on 2 GB

import os
import gc
import json
import math
import time
from typing import List, Dict, Any, Iterable, Tuple

from openai import OpenAI
from psycopg.types.json import Json

# ---------- Config via env ----------
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# Chunking (characters; token-agnostic maar conservatief)
CHUNK_CHARS = int(os.getenv("INGEST_CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("INGEST_CHUNK_OVERLAP", "200"))

# Microbatch grootte voor embedding-calls
EMBED_BATCH = int(os.getenv("INGEST_BATCH", "5"))

# Safeties
MAX_CHUNKS_PER_PAGE = int(os.getenv("INGEST_MAX_CHUNKS_PER_PAGE", "60"))
MAX_TOTAL_CHUNKS = int(os.getenv("INGEST_MAX_TOTAL_CHUNKS", "500"))

# Retry/backoff
RETRY_MAX = 4
RETRY_BASE_SLEEP = 0.8

# OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ---------- Helpers ----------

def _cap_str(s: str, max_len: int) -> str:
    if not s:
        return ""
    if len(s) <= max_len:
        return s
    return s[:max_len]

def _chunk_text(text: str, chunk_chars: int, overlap: int, max_chunks: int) -> Iterable[str]:
    """
    Chunk on characters (fast, token-agnostic).
    Ensures some overlap to keep context continuity.
    Caps total number of chunks for safety.
    """
    text = text or ""
    n = len(text)
    if n == 0:
        return []
    res: List[str] = []
    i = 0
    while i < n and len(res) < max_chunks:
        end = min(i + chunk_chars, n)
        res.append(text[i:end])
        i = end - overlap
        if i <= 0:
            i = end  # avoid stuck if overlap > i
    return res

def _build_page_text(p: Dict[str, Any]) -> str:
    """
    Maak compacte, semantische content uit crawl-page velden.
    Houd het kort per sectie zodat chunks niet onnodig groot worden.
    """
    bits: List[str] = []

    title = p.get("title") or ""
    h1 = p.get("h1") or ""
    meta_desc = p.get("meta_description") or ""

    if title: bits.append(f"TITLE: {_cap_str(title, 400)}")
    if h1: bits.append(f"H1: {_cap_str(h1, 400)}")
    if meta_desc: bits.append(f"DESCRIPTION: {_cap_str(meta_desc, 600)}")

    for h in (p.get("h2") or [])[:20]:
        if h: bits.append(f"H2: {_cap_str(str(h), 300)}")
    for h in (p.get("h3") or [])[:30]:
        if h: bits.append(f"H3: {_cap_str(str(h), 250)}")

    # Canonical/noindex/nofollow kunnen nuttig zijn als labels
    if p.get("canonical"):
        bits.append(f"CANONICAL: {_cap_str(p['canonical'], 500)}")
    if p.get("noindex") is True:
        bits.append("ROBOTS: noindex")
    if p.get("nofollow") is True:
        bits.append("ROBOTS: nofollow")

    return "\n".join(bits).strip()

def _embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Haal embeddings op voor een kleine batch 'texts'.
    Met retries & exponential backoff.
    """
    attempt = 0
    while True:
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            # OpenAI’s volgorde == input-volgorde
            return [d.embedding for d in resp.data]
        except Exception as e:
            attempt += 1
            if attempt > RETRY_MAX:
                raise
            sleep_s = RETRY_BASE_SLEEP * (2 ** (attempt - 1))
            time.sleep(sleep_s)

def _iter_microbatches(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# ---------- Public API ----------

def ingest_crawl_output(conn, site_id: str, crawl_output: Dict[str, Any]) -> int:
    """
    Verwerk crawl-output in kleine, stabiele stappen:
    - Per pagina compacte tekst maken
    - In kleine chunks knippen
    - Embeddings ophalen in microbatches
    - Direct INSERTen in Postgres (vector kolom)
    Houdt RAM stabiel (geen grote arrays vasthouden).
    """
    pages = (crawl_output or {}).get("pages") or []
    if not pages:
        return 0

    inserted_total = 0
    chunks_total = 0

    with conn.cursor() as cur:
        for p in pages:
            # Als we onze totale cap hebben bereikt → stoppen
            if chunks_total >= MAX_TOTAL_CHUNKS:
                break

            url = p.get("final_url") or p.get("url")
            status = p.get("status")
            if not url or status != 200:
                continue

            # Bouw compacte tekst uit de velden
            page_text = _build_page_text(p)
            if not page_text:
                continue

            # Chunk per pagina met cap
            chunks = list(_chunk_text(
                page_text,
                chunk_chars=CHUNK_CHARS,
                overlap=CHUNK_OVERLAP,
                max_chunks=min(MAX_CHUNKS_PER_PAGE, MAX_TOTAL_CHUNKS - chunks_total)
            ))
            if not chunks:
                continue

            # Microbatch embeddings en INSERT direct
            for batch in _iter_microbatches(chunks, EMBED_BATCH):
                # 1) Embeddings (klein batchje)
                embs = _embed_batch(batch)

                # 2) Directe inserts (één per chunk)
                for text_chunk, emb in zip(batch, embs):
                    # NB: pgvector accepteert Python list -> vector
                    cur.execute("""
                        INSERT INTO documents (site_id, url, language, content, metadata, embedding)
                        VALUES (%s, %s, NULL, %s, %s, %s)
                    """, (site_id, url, text_chunk, Json({"source":"crawl"}), emb))
                    inserted_total += 1
                    chunks_total += 1
                    if chunks_total >= MAX_TOTAL_CHUNKS:
                        break

                # 3) Flush & GC per microbatch
                conn.commit()
                del embs
                gc.collect()

                if chunks_total >= MAX_TOTAL_CHUNKS:
                    break

            # Zwakke hint aan GC na pagina
            del chunks, page_text
            gc.collect()

    return inserted_total
