import os, psycopg, json
from rag_helper import embed

DATABASE_URL = os.environ["DATABASE_URL"]

docs = [
  {
    "source": "Schema.org",
    "title": "FAQPage best practices",
    "url": "https://schema.org/FAQPage",
    "tags": ["Schema","AEO","FAQ"],
    "content": "FAQPage in JSON-LD. Each Question has acceptedAnswer. Keep answers concise; avoid promotional language."
  },
  {
    "source": "Search Central",
    "title": "Structured data basics",
    "url": "https://developers.google.com/search/docs/appearance/structured-data",
    "tags": ["Schema","SEO"],
    "content": "Structured data helps search engines understand content. Use JSON-LD, keep it consistent with visible content."
  },
  {
    "source": "Entity linking",
    "title": "sameAs & entity grounding",
    "url": "https://example.org/entities",
    "tags": ["Entities","AEO"],
    "content": "Use sameAs to link brand/org to authoritative IDs (Wikipedia, Wikidata, LinkedIn). Improves disambiguation in LLMs."
  }
]

with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
    for d in docs:
        vec = embed(d["content"])
        cur.execute("""
            INSERT INTO kb_documents (source,url,title,tags,content,embedding)
            VALUES (%s,%s,%s,%s,%s,%s)
        """, (d["source"], d["url"], d["title"], d["tags"], d["content"], vec))
    conn.commit()

print("KB seeded:", len(docs))
