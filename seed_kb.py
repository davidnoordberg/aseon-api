import os, sys, json, hashlib
from typing import List, Dict, Any
import psycopg
from openai import OpenAI

DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

client = OpenAI(api_key=OPENAI_API_KEY)

def _hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

def _embed(text: str):
    resp = client.embeddings.create(model=EMBED_MODEL, input=[(text or "").strip()])
    return resp.data[0].embedding

def _load_yaml(path: str) -> List[Dict[str, Any]]:
    try:
        import yaml  # type: ignore
    except Exception:
        print("PyYAML niet gevonden. Installeer met: pip install pyyaml", file=sys.stderr)
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    if isinstance(data, dict) and "docs" in data:
        return data["docs"]
    if isinstance(data, list):
        return data
    return []

FALLBACK_DOCS = [
    {
        "source": "Schema.org",
        "title": "FAQPage essentials",
        "url": "https://schema.org/FAQPage",
        "tags": ["Schema","AEO","FAQ"],
        "content": (
            "Use JSON-LD FAQPage. Each Question must have an acceptedAnswer with concise, factual text. "
            "Answers should reflect visible content. Avoid promotional statements."
        ),
    },
    {
        "source": "Search Central",
        "title": "Structured data basics",
        "url": "https://developers.google.com/search/docs/appearance/structured-data",
        "tags": ["Schema","SEO"],
        "content": (
            "Structured data helps search engines understand content. Prefer JSON-LD. "
            "Content in markup must match the visible page. Keep it accurate and updated."
        ),
    },
    {
        "source": "AEO Guidelines",
        "title": "Citation-readiness for LLMs",
        "url": "https://example.org/llm-citation-readiness",
        "tags": ["AEO","Citations"],
        "content": (
            "Create short, self-contained answers (≤80 words) with a canonical URL. "
            "Provide primary sources and stable entity references (sameAs) to improve citation likelihood."
        ),
    },
    {
        "source": "Entities",
        "title": "Entity grounding with sameAs",
        "url": "https://example.org/entities",
        "tags": ["Entities","AEO"],
        "content": (
            "Link brands, people, and products to authoritative IDs using sameAs (Wikipedia, Wikidata, LinkedIn). "
            "Use about/mentions in Article to reinforce entity disambiguation."
        ),
    },
    {
        "source": "Technical SEO",
        "title": "Crawl basics",
        "url": "https://example.org/crawl-basics",
        "tags": ["SEO"],
        "content": (
            "Ensure valid 200 responses for canonical pages, one H1 per page, unique titles/meta descriptions, "
            "and accurate rel=canonical. Avoid noindex on indexable pages."
        ),
    },
    {
        "source": "Schema.org",
        "title": "Organization essentials",
        "url": "https://schema.org/Organization",
        "tags": ["Schema","Entities"],
        "content": (
            "Organization should include name, url, logo, contact info, and sameAs links to authoritative profiles. "
            "Use LocalBusiness subtype when applicable."
        ),
    },
    {
        "source": "Schema.org",
        "title": "Article essentials",
        "url": "https://schema.org/Article",
        "tags": ["Schema"],
        "content": (
            "Article should include headline, author, datePublished, mainEntityOfPage and, where relevant, "
            "about/mentions of key entities. Keep markup consistent with on-page content."
        ),
    },
    {
        "source": "Navigation",
        "title": "BreadcrumbList",
        "url": "https://schema.org/BreadcrumbList",
        "tags": ["Schema","SEO"],
        "content": (
            "BreadcrumbList improves understanding of site hierarchy. Use itemListElement with ListItem entries "
            "including name and item URL in page templates."
        ),
    },
    {
        "source": "Answer Fragments",
        "title": "Creating high-utility FAQs",
        "url": "https://example.org/answer-fragments",
        "tags": ["AEO","FAQ"],
        "content": (
            "Write FAQs that answer one clear question each, ≤80 words, with internal links or anchors to deeper pages. "
            "Avoid generic marketing language; be specific and factual."
        ),
    },
    {
        "source": "Performance",
        "title": "Page speed and AEO",
        "url": "https://example.org/aeo-speed",
        "tags": ["AEO","SEO"],
        "content": (
            "Slow pages reduce crawl coverage and user trust. Optimize CLS/LCP/INP and keep HTML lightweight to "
            "help both search engines and LLM retrievers."
        ),
    },
]

def main():
    path = os.path.join(os.getcwd(), "kb_seed.yaml")
    from_yaml = _load_yaml(path) if os.path.exists(path) else []
    docs = from_yaml if from_yaml else FALLBACK_DOCS
    print(f"Seeding KB with {len(docs)} docs")

    with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        inserted = 0; skipped = 0
        for d in docs:
            content = (d.get("content") or "").strip()
            if not content:
                skipped += 1
                continue
            chash = _hash(content)
            vec = _embed(content)
            cur.execute("""
                INSERT INTO kb_documents (source,url,title,tags,content,embedding,content_hash)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (url, content_hash) DO NOTHING
            """, (
                d.get("source"),
                d.get("url"),
                d.get("title"),
                d.get("tags") or [],
                content,
                vec,
                chash
            ))
            if cur.rowcount > 0:
                inserted += 1
            else:
                skipped += 1
        conn.commit()
    print(json.dumps({"inserted": inserted, "skipped": skipped}))

if __name__ == "__main__":
    main()
