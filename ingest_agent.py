# ingest_agent.py
import os, json
from typing import List, Dict, Any
from openai import OpenAI

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def _chunk_text(text: str, max_chars: int = 1500, overlap: int = 150) -> List[str]:
    chunks, i, n = [], 0, len(text)
    while i < n:
        end = min(i + max_chars, n)
        chunks.append(text[i:end])
        i = end - overlap
        if i < 0: i = 0
        if i >= n: break
    return chunks

def _page_snippet(p: Dict[str, Any]) -> str:
    bits = []
    for k in ("title","h1","meta_description"):
        v = p.get(k)
        if v: bits.append(str(v))
    for h in (p.get("h2") or [])[:5]: bits.append(h)
    for h in (p.get("h3") or [])[:5]: bits.append(h)
    # (optioneel) 1–2 korte paragrafen uit body_text indien aanwezig
    for para in (p.get("paragraphs") or [])[:2]:
        if para and isinstance(para, str):
            bits.append(para.strip())
    return "\n".join([b for b in bits if b]).strip()

def ingest_crawl_output(conn, site_id: str, crawl_output: Dict[str, Any]) -> int:
    pages = (crawl_output or {}).get("pages") or []
    if not pages: return 0
    inserted = 0
    with conn.cursor() as cur:
        for p in pages:
            url = p.get("final_url") or p.get("url")
            if not url: 
                continue
            content = _page_snippet(p)
            if not content:
                continue
            for chunk in _chunk_text(content):
                emb = client.embeddings.create(model=EMBED_MODEL, input=[chunk]).data[0].embedding
                cur.execute("""
                    INSERT INTO documents (site_id, url, language, content, metadata, embedding)
                    VALUES (%s, %s, NULL, %s, %s, %s)
                """, (site_id, url, chunk, json.dumps({"source":"crawl"}), emb))
                inserted += 1
        conn.commit()
    return inserted
