# schema_agent.py
import os, json
from urllib.parse import urlparse
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

_BASE_SYS = """
You are an expert Schema.org JSON-LD generator for SEO/AEO.
Produce PERFECT, production-ready structured data.

Hard rules (must follow):
- Return a single valid JSON object (no arrays), no comments/explanations.
- Always include "@context":"https://schema.org".
- Only emit fields you can infer from the input (domain/name) or that are explicitly provided.
- Do NOT fabricate phone numbers, addresses, coordinates, prices, ratings, or opening hours.
- Respect the requested language if provided (field values should be in that language).
- Keep any FAQ answers concise (≤80 words), factual, and non-promotional.

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

Output must be compact and valid JSON-LD.
"""

def _call_llm(prompt: str) -> dict | None:
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
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception:
        # Fallback zonder response_format als de modelrespons plain JSON is
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": _BASE_SYS},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            content = resp.choices[0].message.content
            return json.loads(content)
        except Exception as e2:
            print(json.dumps({
                "level": "ERROR",
                "msg": "schema_llm_failed",
                "error": str(e2)
            }), flush=True)
            return None

def _fallback_schema(biz_type: str, site_name: str, site_url: str) -> dict:
    return {
        "@context": "https://schema.org",
        "@type": biz_type,
        "name": site_name,
        "url": site_url
    }

def validate_schema(data: dict, biz_type: str) -> tuple[bool, str | None]:
    if not isinstance(data, dict):
        return False, "Schema is not a dict"
    if "@type" not in data:
        return False, "Missing @type"
    t = data.get("@type")
    if t in ("Organization", "LocalBusiness"):
        if not data.get("name") or not data.get("url"):
            return False, f"{t} missing name or url"
    if t == "Article":
        for k in ("headline", "description", "author", "datePublished", "mainEntityOfPage", "image"):
            if not data.get(k):
                return False, f"Article missing {k}"
    if t == "FAQPage":
        me = data.get("mainEntity")
        if not me or not isinstance(me, list):
            return False, "FAQPage missing mainEntity"
        first = me[0] if me else {}
        aa = (first or {}).get("acceptedAnswer")
        if not aa or not isinstance(aa, dict) or not aa.get("text"):
            return False, "FAQPage missing acceptedAnswer"
    return True, None

def generate_schema(
    biz_type: str,
    site_name: str | None,
    site_url: str,
    language: str | None = None,
    extras: dict | None = None,
    rag_context: str | None = None
) -> dict:
    """
    Genereer JSON-LD voor het opgegeven type. Geen verzonnen data.
    'rag_context' kan crawl- of andere samenvattingen bevatten ter onderbouwing.
    """
    extras = extras or {}
    bt = (biz_type or "Organization").strip()
    domain_name = urlparse(site_url).netloc
    name = (site_name or domain_name).strip()

    payload = {
        "biz_type": bt,
        "defaults": {
            "name": name,
            "url": site_url,
            "language": language
        },
        "extras": extras,
        "context": rag_context or "(no extra site context)"
    }

    prompt = (
        "Generate a single JSON-LD object that follows the rules. "
        "Only include fields you can infer from this input/context (do not fabricate contact details):\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )

    data = _call_llm(prompt)

    if not isinstance(data, dict) or not data:
        data = _fallback_schema(bt, name, site_url)

    data.setdefault("@context", "https://schema.org")
    data.setdefault("@type", bt)

    ok, err = validate_schema(data, bt)
    if not ok:
        print(json.dumps({
            "level": "WARN",
            "msg": "schema_invalid",
            "error": err
        }), flush=True)
        data = _fallback_schema(bt, name, site_url)

    return data
