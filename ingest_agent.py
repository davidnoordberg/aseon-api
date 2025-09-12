# ingest_agent.py
import os, json
from typing import List, Dict, Any
import psycopg
from psycopg.rows import dict_row
from openai import OpenAI

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def _chunk_text(text: str, max_chars: int = 1800, overlap: int = 200) -> List[str]:
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + max_chars, n)
        chunk = text[i:end]
        chunks.append(chunk)
        i = end - overlap
        if i < 0: i = 0
        if i >= n: break
    return chunks

def ingest_crawl_output(conn, site_id: str, crawl_output: Dict[str, Any]) -> int:
    pages = (crawl_output or {}).get("pages") or []
    inserted = 0
    with conn.cursor() as cur:
        for p in pages:
            url = p.get("url")
            if not url: 
                continue
            # build content from title/h1/h2/h3/meta
            bits = []
            for k in ("title","h1","meta_description"):
                v = p.get(k)
                if v: bits.append(str(v))
            for h in (p.get("h2") or []): bits.append(h)
            for h in (p.get("h3") or []): bits.append(h)
            content = "\n".join(bits).strip()
            if not content:
                continue
            for chunk in _chunk_text(content):
                emb = client.embeddings.create(model=EMBED_MODEL, input=[chunk]).data[0].embedding
                vec = emb  # psycopg3 can pass list to vector if extension supports cast
                cur.execute("""
                    INSERT INTO documents (site_id, url, language, content, metadata, embedding)
                    VALUES (%s, %s, NULL, %s, %s, %s)
                """, (site_id, url, chunk, json.dumps({"source":"crawl"}), vec))
                inserted += 1
        conn.commit()
    return inserted
