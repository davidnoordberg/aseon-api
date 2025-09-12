# schema_agent.py
import os, json
from urllib.parse import urlparse
from typing import Optional, Tuple, Dict, Any
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

_BASE_SYS = """
You are an expert Schema.org JSON-LD generator for SEO/AEO.
Produce PERFECT, production-ready structured data.

Hard rules (must follow):
- Return a single valid JSON object (no arrays), no comments/explanations.
- Always include "@context":"https://schema.org".
- Only emit fields you can infer from the input/context or that are explicitly provided.
- Do NOT fabricate phone numbers, addresses, geo coordinates, opening hours, prices, ratings, or reviews.
- Respect the requested language if provided (field values should be in that language).
- Keep any FAQ answers concise (≤80 words), factual, non-promotional.

Supported types and minimum fields:
1) Organization / LocalBusiness:
   - "@type": "Organization" or "LocalBusiness"
   - name, url
   - Optional when known: logo, sameAs[], address, telephone
2) Article:
   - headline, description, author { name, url }, datePublished, mainEntityOfPage, image
3) FAQPage:
   - mainEntity: [ { "@type": "Question", "name": "...", "acceptedAnswer": { "@type": "Answer", "text": "..." } } ]
4) OfferCatalog / Product:
   - name, description, url
   - itemListElement[] for OfferCatalog if items present

Output must be compact and valid JSON-LD. If context is insufficient to safely fill a field, omit it.
"""

def _call_llm(prompt: str) -> Optional[Dict[str, Any]]:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _BASE_SYS},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": _BASE_SYS},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception:
            return None

def _fallback_schema(biz_type: str, site_name: str, site_url: str) -> Dict[str, Any]:
    return {"@context": "https://schema.org", "@type": biz_type, "name": site_name, "url": site_url}

def validate_schema(data: Dict[str, Any], biz_type: str) -> Tuple[bool, Optional[str]]:
    if not isinstance(data, dict): return False, "Schema is not a dict"
    if "@type" not in data: return False, "Missing @type"
    t = data.get("@type")
    if t in ("Organization","LocalBusiness"):
        if not data.get("name") or not data.get("url"): return False, f"{t} missing name or url"
    if t == "Article":
        for k in ("headline","description","author","datePublished","mainEntityOfPage","image"):
            if not data.get(k): return False, f"Article missing {k}"
    if t == "FAQPage":
        me = data.get("mainEntity")
        if not me or not isinstance(me, list): return False, "FAQPage missing mainEntity"
        first = me[0] if me else {}
        aa = (first or {}).get("acceptedAnswer")
        if not aa or not isinstance(aa, dict) or not aa.get("text"):
            return False, "FAQPage missing acceptedAnswer"
    return True, None

def generate_schema(
    biz_type: str,
    site_name: Optional[str],
    site_url: str,
    language: Optional[str] = None,
    extras: Optional[Dict[str, Any]] = None,
    rag_context: Optional[str] = None
) -> Dict[str, Any]:
    extras = extras or {}
    bt = (biz_type or "Organization").strip()
    domain_name = urlparse(site_url).netloc
    name = (site_name or domain_name).strip()

    payload = {
        "biz_type": bt,
        "defaults": {"name": name, "url": site_url, "language": language},
        "extras": extras,
        "context": rag_context or "(no extra site context)"
    }

    prompt = (
        "Generate ONE JSON-LD object that follows the rules. "
        "Use the provided context when helpful. "
        "Only include fields you can safely infer; do NOT fabricate contact info.\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )

    data = _call_llm(prompt)
    if not isinstance(data, dict) or not data:
        data = _fallback_schema(bt, name, site_url)

    data.setdefault("@context", "https://schema.org")
    data.setdefault("@type", bt)

    ok, err = validate_schema(data, bt)
    if not ok:
        data = _fallback_schema(bt, name, site_url)

    # merge some extras (safe keys)
    for k in ("sameAs","logo"):
        if k in (extras or {}) and k not in data:
            data[k] = extras[k]

    return data
