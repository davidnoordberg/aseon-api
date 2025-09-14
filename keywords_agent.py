# keywords_agent.py
import os, json, re
from typing import Dict, Any, List, Tuple
import psycopg
from psycopg.rows import dict_row
from openai import OpenAI

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "20"))

MAX_CTX_CHARS = int(os.getenv("KW_MAX_CTX_CHARS", "4000"))
DOC_K = int(os.getenv("KW_DOC_TOPK", "8"))
KB_K  = int(os.getenv("KW_KB_TOPK", "5"))

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def _embed(text: str) -> List[float]:
    text = (text or "").strip()
    return client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding

def _rows_to_ctx(rows: List[dict], label: str) -> str:
    bits: List[str] = []
    for i, r in enumerate(rows):
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip()
        content = (r.get("content") or "").strip()
        pre = f"[{label.upper()} {i+1}] {url}"
        if title: pre += f" • {title}"
        snippet = (content[:1200] + "…") if len(content) > 1200 else content
        if snippet:
            bits.append(pre + "\n" + snippet)
    return "\n\n".join(bits)

def _query_site_documents(conn, site_id: str, seed: str, k: int) -> List[dict]:
    try:
        qvec = _embed(seed)
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT url, title, content
                  FROM documents
                 WHERE site_id = %s
                 ORDER BY embedding <-> %s::vector
                 LIMIT %s
            """, (site_id, qvec, k))
            return cur.fetchall() or []
    except Exception as e:
        try: conn.rollback()
        except Exception: pass
        print(json.dumps({"level":"WARN","msg":"kw_site_vector_failed","error":str(e)[:300]}), flush=True)
        return []

def _query_kb(conn, seed: str, k: int) -> List[dict]:
    try:
        qvec = _embed(seed)
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT url, title, content
                  FROM kb_documents
                 ORDER BY embedding <-> %s::vector
                 LIMIT %s
            """, (qvec, k))
            return cur.fetchall() or []
    except Exception as e:
        try: conn.rollback()
        except Exception: pass
        print(json.dumps({"level":"WARN","msg":"kw_kb_vector_failed","error":str(e)[:300]}), flush=True)
        return []

def _fallback_crawl_snapshot(conn, site_id: str, max_pages: int = 8) -> str:
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
            h2 = " | ".join(p.get("h2") or [])
            h3 = " | ".join(p.get("h3") or [])
            meta = p.get("meta_description") or ""
            paras = " ".join((p.get("paragraphs") or [])[:2])
            snippet = "\n".join([x for x in [url, title, h1, h2, h3, meta, paras] if x])
            if snippet.strip():
                bits.append(snippet)
        return "\n\n".join(bits).strip()
    except Exception as e:
        try: conn.rollback()
        except Exception: pass
        print(json.dumps({"level":"WARN","msg":"kw_crawl_ctx_failed","error":str(e)[:300]}), flush=True)
        return ""

def _trim_ctx(*chunks: Tuple[str, str]) -> Tuple[str, List[str]]:
    used: List[str] = []
    buf: List[str] = []
    total = 0
    labels_map = {"SITE":"documents","KB":"kb","CRAWL":"crawl"}
    for label, text in chunks:
        if not text: continue
        t = text.strip()
        if not t: continue
        add = len(t)
        if total + add > MAX_CTX_CHARS:
            t = t[: max(0, MAX_CTX_CHARS - total)]
            add = len(t)
        if add <= 0: break
        buf.append(t)
        total += add
        if label in labels_map and labels_map[label] not in used:
            used.append(labels_map[label])
        if total >= MAX_CTX_CHARS: break
    return ("\n\n".join(buf), used)

def generate_keywords(conn, site_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    seed = (payload or {}).get("seed") or "site"
    n = int((payload or {}).get("n", 30))
    market = (payload or {}).get("market", {}) or {}
    language = market.get("language") or "en"
    country  = market.get("country")  or "NL"

    # Context ophalen
    site_rows = _query_site_documents(conn, site_id, seed, DOC_K)
    kb_rows   = _query_kb(conn, seed, KB_K)
    site_ctx  = _rows_to_ctx(site_rows, "site")
    kb_ctx    = _rows_to_ctx(kb_rows,   "kb")
    crawl_ctx = _fallback_crawl_snapshot(conn, site_id, max_pages=8)

    ctx_text, ctx_used = _trim_ctx(
        ("SITE", site_ctx),
        ("KB",   kb_ctx),
        ("CRAWL", crawl_ctx),
    )

    # SYSTEM PROMPT (GEO + AEO bewust)
    sys = f"""
You are an SEO+GEO strategist. Use ONLY the provided site context and KB excerpts.
Target market: {country} ({language}).
Goals:
- Propose realistic queries people would use (web search + assistant prompts).
- Favor topics/entities present in the context.
- Include queries that lead to **citable, answer-first sections** (GEO/AEO).
Output JSON only:
{{
  "keywords": [ "...", ... ],
  "clusters": {{
    "informational": [...],
    "transactional": [...],
    "navigational": [...]
  }},
  "suggestions": [
    {{ "page_title": "...", "grouped_keywords": ["...","..."], "notes": "what to answer first and what to prove" }}
  ]
}}
Constraints:
- 20–50 total keywords.
- Keep on-topic; no generic filler.
- Prefer language/terms seen in context.
""".strip()

    user = f"Seed/topic: {seed}\n\nContext:\n{ctx_text or '(no context)'}"

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        response_format={"type":"json_object"},
        temperature=0.35,
        timeout=OPENAI_TIMEOUT_SEC,
    )

    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        data = {"keywords": [], "clusters": {"informational":[],"transactional":[],"navigational":[]}, "suggestions": []}

    # ban off-topic classes
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

    # Trim window 20–50
    if len(kws) < 20:
        extra = (clusters["informational"] + clusters["transactional"] + clusters["navigational"])
        for k in extra:
            if k not in kws:
                kws.append(k)
            if len(kws) >= max(20, n): break
    if len(kws) > 50:
        kws = kws[:50]

    suggestions = data.get("suggestions") or []
    out = {
        "seed": seed,
        "language": language,
        "country": country,
        "source_context": (ctx_used[0] if ctx_used else None),  # for backward compat
        "_context_used": (ctx_used or None),
        "keywords": kws,
        "clusters": clusters,
        "suggestions": suggestions
    }
    print(json.dumps({"level":"INFO","msg":"keywords_generated","context_used":ctx_used,"n":len(kws)}), flush=True)
    return out
