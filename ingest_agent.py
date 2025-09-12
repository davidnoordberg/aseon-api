# ingest_agent.py
import os, json, time
from typing import List, Dict, Any
from openai import OpenAI
from psycopg.rows import dict_row

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "8"))
EMBED_SLEEP_SEC = float(os.getenv("EMBED_SLEEP_SEC", "0.1"))

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def _chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    chunks, i, n = [], 0, len(text or "")
    while i < n:
        end = min(i + max_chars, n)
        chunk = text[i:end].strip()
        if chunk: chunks.append(chunk)
        i = max(end - overlap, end)  # move forward, keep small overlap
    return chunks

def _page_to_textbits(p: Dict[str, Any]) -> List[str]:
    bits = []
    for k in ("title","h1","meta_description"):
        v = p.get(k)
        if v: bits.append(str(v))
    for h in (p.get("h2") or []): bits.append(h)
    for h in (p.get("h3") or []): bits.append(h)
    # voeg korte body-paragrafen toe (beperkt)
    for para in (p.get("paragraphs") or [])[:2]:
        bits.append(para)
    return [b for b in (bits or []) if isinstance(b, str) and b.strip()]

def ingest_crawl_output(conn, site_id: str, crawl_output: Dict[str, Any]) -> int:
    pages = (crawl_output or {}).get("pages") or []
    to_embed: List[Dict[str, Any]] = []

    for p in pages:
        url = p.get("final_url") or p.get("url")
        if not url: continue
        bits = _page_to_textbits(p)
        if not bits: continue
        content = "\n".join(bits).strip()
        if not content: continue
        for chunk in _chunk_text(content, max_chars=900, overlap=120):
            to_embed.append({"url": url, "text": chunk})

    inserted = 0
    if not to_embed:
        return inserted

    with conn.cursor(row_factory=dict_row) as cur:
        # batch embed + insert
        for i in range(0, len(to_embed), EMBED_BATCH_SIZE):
            batch = to_embed[i:i+EMBED_BATCH_SIZE]
            texts = [b["text"] for b in batch]

            # OpenAI embeddings (batched)
            emb_resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            vectors = [item.embedding for item in emb_resp.data]

            for rec, vec in zip(batch, vectors):
                cur.execute("""
                    INSERT INTO documents (site_id, url, language, content, metadata, embedding)
                    VALUES (%s, %s, NULL, %s, %s, %s)
                """, (site_id, rec["url"], rec["text"], json.dumps({"source":"crawl"}), vec))
                inserted += 1
            conn.commit()
            time.sleep(EMBED_SLEEP_SEC)
    return inserted
