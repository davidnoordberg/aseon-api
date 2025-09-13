# keywords_agent.py
import os, json, re
from typing import Dict, Any, List, Tuple, Optional
import psycopg
from psycopg.rows import dict_row
from openai import OpenAI

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "20"))

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ---------------- Embedding helper ----------------

def _embed(text: str) -> List[float]:
    text = (text or "").strip()
    if not text:
        return client.embeddings.create(model=EMBED_MODEL, input=[""]).data[0].embedding
    return client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding

# ---------------- Context builders ----------------

def _context_from_documents(conn, site_id: str, seed: str, k: int = 8) -> Tuple[str, str]:
    """
    Haal top-k documenten via pgvector. Vereist: documents.embedding = VECTOR(1536)
    """
    try:
        qvec = _embed(seed)
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT url, content
                  FROM documents
                 WHERE site_id = %s
                 ORDER BY embedding <-> %s::vector
                 LIMIT %s
            """, (site_id, qvec, k))
            rows = cur.fetchall()
        if rows:
            ctx = "\n\n".join([f"[{i+1}] {r['url']}\n{(r['content'] or '')}" for i, r in enumerate(rows)])
            return ctx, "documents"
    except Exception as e:
        try: conn.rollback()
        except Exception: pass
        print(json.dumps({"level":"WARN","msg":"keywords_rag_failed","error":str(e)[:300]}), flush=True)
    return "", "none"

def _context_from_latest_crawl(conn, site_id: str, max_pages: int = 8) -> Tuple[str, str]:
    """
    Fallback: gebruik de laatste crawl job output (titels, h1, h2, meta, 1-2 paragrafen).
    """
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT output
                  FROM jobs
                 WHERE site_id=%s AND type='crawl' AND status='done'
              ORDER BY COALESCE(finished_at, created_at) DESC
                 LIMIT 1
            """, (site_id,))
            r = cur.fetchone()
        out = (r or {}).get("output") or {}
        pages = out.get("pages") or []
        bits: List[str] = []
        for p in pages[:max_pages]:
            url = p.get("final_url") or p.get("url") or ""
            title = p.get("title") or ""
            h1 = p.get("h1") or ""
            h2s = " | ".join(p.get("h2") or [])
            h3s = " | ".join(p.get("h3") or [])
            meta = p.get("meta_description") or ""
            paras = " ".join((p.get("paragraphs") or [])[:2])
            snippet = "\n".join([x for x in [url, title, h1, h2s, h3s, meta, paras] if x])
            if snippet.strip():
                bits.append(snippet)
        ctx = "\n\n".join(bits).strip()
        return (ctx, "crawl") if ctx else ("", "none")
    except Exception as e:
        try: conn.rollback()
        except Exception: pass
        print(json.dumps({"level":"WARN","msg":"keywords_crawl_ctx_failed","error":str(e)[:300]}), flush=True)
        return "", "none"

# ---------------- Main generator ----------------

def generate_keywords(conn, site_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nieuwe RAG-gedreven keywords generator.
    - 1) Probeer RAG over documents (pgvector)
    - 2) Fallback: context uit laatste crawl output
    - 3) Laatste fallback: seed-only (minimaal, maar nooit generieke troep als 'home improvement')
    """
    seed = (payload or {}).get("seed") or "site"
    n = int((payload or {}).get("n", 30))
    market = (payload or {}).get("market", {})
    language = market.get("language") or "en"
    country  = market.get("country")  or "NL"

    # 1) RAG uit documents
    ctx, ctx_label = _context_from_documents(conn, site_id, seed, k=10)

    # 2) Fallback: laatste crawl-context
    if not ctx:
        crawl_ctx, crawl_label = _context_from_latest_crawl(conn, site_id, max_pages=8)
        if crawl_ctx:
            ctx, ctx_label = crawl_ctx, crawl_label

    # 3) Prompt
    sys = f"""
You are an SEO strategist for a SaaS in the Generative/Answer/AI visibility space (SEO+AEO+GEO).
Use ONLY the provided site context (content and headings) to propose realistic search queries people in {country} ({language}) would use.
Rules:
- Return exactly these fields as JSON: {{
  "keywords": [ "...", ... ],
  "clusters": {{
    "informational": [...],
    "transactional": [...],
    "navigational": [...]
  }},
  "suggestions": [
    {{
      "page_title": "...",
      "grouped_keywords": ["...", "..."]
    }}
  ]
}}
- No generic 'home improvement' or irrelevant topics. Keep it on-topic for the site's content.
- Prefer queries that match the terminology and entities found in the context.
- 20–50 total keywords.
    """.strip()

    user = f"Seed/topic: {seed}\n\nContext:\n{(ctx if ctx else '(no context)')}"

    # Als echt nul context, geef dan een duidelijke guardrail
    if not ctx:
        sys += "\nIf there is no context, generate a minimal, generic set about Generative SEO / AEO / GEO SaaS, not household/home topics."

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        response_format={"type": "json_object"},
        temperature=0.4,
        timeout=OPENAI_TIMEOUT_SEC,
    )

    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        data = {"keywords": [], "clusters": {"informational":[],"transactional":[],"navigational":[]}, "suggestions": []}

    # sanity: hard filter hele off-topic families (home improvement e.d.)
    ban = re.compile(r"\b(home improvement|furniture|appliance|garden|decor|warranty|loan|sell a house|real estate)\b", re.I)
    def filt(xs: List[str]) -> List[str]:
        out = []
        for x in xs or []:
            x = (x or "").strip()
            if not x or ban.search(x): 
                continue
            out.append(x)
        return out

    kws = filt(data.get("keywords") or [])
    clusters = data.get("clusters") or {}
    clusters = {
        "informational": filt(clusters.get("informational") or []),
        "transactional": filt(clusters.get("transactional") or []),
        "navigational": filt(clusters.get("navigational") or []),
    }

    # Trim/limit totals
    if len(kws) < 10 and ctx:  # als we context hadden en toch weinig, dupliceer uit clusters
        extra = (clusters["informational"] + clusters["transactional"] + clusters["navigational"])[: max(0, n - len(kws))]
        kws = list(dict.fromkeys(kws + extra))
    kws = kws[:max(n, 20)]  # 20–50 window
    if len(kws) > 50:
        kws = kws[:50]

    suggestions = data.get("suggestions") or []
    out = {
        "seed": seed,
        "language": language,
        "country": country,
        "source_context": ctx_label if ctx_label != "none" else None,
        "keywords": kws,
        "clusters": clusters,
        "suggestions": suggestions
    }
    print(json.dumps({"level":"INFO","msg":"keywords_generated","source_context":out["source_context"],"n":len(kws)}), flush=True)
    return out
