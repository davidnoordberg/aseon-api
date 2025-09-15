# keywords_agent.py
import os
import json
import re
from typing import Dict, Any, List
from openai import OpenAI

from rag_helper import search_site_docs, search_kb, build_context

CHAT_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "20"))
MAX_CTX_CHARS = int(os.getenv("KW_MAX_CTX_CHARS", "4000"))

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _dedupe_keep_order(xs: List[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        k = (x or "").strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append((x or "").strip())
    return out


def generate_keywords(conn, site_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    seed = (payload or {}).get("seed") or "site"
    n = int((payload or {}).get("n", 30))
    market = (payload or {}).get("market", {}) or {}
    language = (market.get("language") or "en").lower()
    country  = (market.get("country")  or "NL").upper()

    site_rows = search_site_docs(conn, site_id, seed, k=8)
    kb_rows   = search_kb(conn, seed, k=6, tags=["Schema","SEO","AEO","Content","Quality"])
    ctx = build_context(site_rows, kb_rows, budget_chars=MAX_CTX_CHARS)

    system = f"""
You are an SEO+GEO strategist for {country} ({language}).
Use ONLY the provided context. Return JSON only.
Structure:
{{
  "keywords": ["..."],                       // 20–50 total
  "clusters": {{
    "informational": ["..."],
    "transactional": ["..."],
    "navigational": ["..."]
  }},
  "suggestions": [
    {{"page_title":"...","grouped_keywords":["...","..."],"notes":"answer-first outline & evidence to include"}}
  ]
}}
Rules:
- Mix classic SEO queries and assistant-style prompts.
- Prefer entities/terms present in the site context.
- Keep on-topic; no generic or off-vertical queries.
""".strip()

    user = f"""Seed/topic: {seed}

--- SITE CONTEXT ---
{ctx.get("site_ctx")}

--- KB CONTEXT ---
{ctx.get("kb_ctx")}
"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.35,
        timeout=OPENAI_TIMEOUT_SEC,
        response_format={"type":"json_object"},
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
    )

    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        data = {"keywords": [], "clusters": {"informational":[],"transactional":[],"navigational":[]}, "suggestions": []}

    ban = re.compile(r"\b(home improvement|furniture|appliance|garden|decor|warranty|loan|sell a house|real estate|mortgage)\b", re.I)

    def filt(xs: List[str]) -> List[str]:
        out = []
        for x in xs or []:
            s = (x or "").strip()
            if not s or ban.search(s):
                continue
            out.append(s)
        return _dedupe_keep_order(out)

    kws = filt(data.get("keywords") or [])
    clusters_in = data.get("clusters") or {}
    clusters = {
        "informational": filt(clusters_in.get("informational") or []),
        "transactional": filt(clusters_in.get("transactional") or []),
        "navigational":  filt(clusters_in.get("navigational")  or []),
    }

    pool = clusters["informational"] + clusters["transactional"] + clusters["navigational"]
    for k in pool:
        if len(kws) >= max(20, n):
            break
        if k not in kws:
            kws.append(k)
    if len(kws) > 50:
        kws = kws[:50]

    out = {
        "seed": seed,
        "language": language,
        "country": country,
        "_context_used": {
            "site_citations": ctx.get("site_citations"),
            "kb_citations": ctx.get("kb_citations"),
            "char_used": ctx.get("char_used"),
        },
        "keywords": kws,
        "clusters": clusters,
        "suggestions": data.get("suggestions") or []
    }
    print(json.dumps({"level":"INFO","msg":"keywords_generated","n":len(kws),"seed":seed}), flush=True)
    return out
