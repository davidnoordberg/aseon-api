# schema_agent.py
import os, json
from urllib.parse import urlparse
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Vast instructieblok: SEO, AEO, GEO best practices
_BASE_SYS = """
You are an expert Schema.org JSON-LD generator.
Your job is to produce PERFECT, production-ready structured data.

Rules:
- Always return a single valid JSON object, no comments or explanations.
- Always include "@context": "https://schema.org".
- Output must strictly conform to Schema.org types.

Types you must support:
1. Organization / LocalBusiness
   - Fields: name, url, logo?, sameAs[], address?, telephone?
   - GEO: link entities with sameAs (LinkedIn, Wikidata, socials)
2. Article
   - Fields: headline, description, author{name,url}, datePublished, mainEntityOfPage, image
3. FAQPage
   - Fields: mainEntity: [Question + acceptedAnswer{text}]
   - AEO: Answers must be ≤80 words, factual, no marketing fluff
   - Google: 2–3 FAQ items recommended for rich results
4. OfferCatalog / Product
   - Fields: name, description, url, itemListElement[]
   - Products/offers must have name + url at minimum

General rules:
- Do not fabricate data (phone numbers, addresses, emails). If unknown, omit.
- Language: match the site’s language if given.
- Entities: use sameAs with absolute URLs to authoritative sources.
- Ensure Google Rich Results eligibility: required fields must always be present.
- Validate JSON before returning: must be parseable.
"""

def _call_llm(prompt: str) -> dict:
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
    try:
        return json.loads(content)
    except Exception:
        return {}

def _fallback_schema(biz_type: str, site_name: str, site_url: str) -> dict:
    """Minimal fallback if model fails"""
    return {
        "@context": "https://schema.org",
        "@type": biz_type,
        "name": site_name,
        "url": site_url
    }

def validate_schema(data: dict, biz_type: str) -> tuple[bool, str | None]:
    """Lightweight validator: checks required fields per type"""
    if not isinstance(data, dict):
        return False, "Schema is not a dict"
    if "@type" not in data:
        return False, "Missing @type"
    t = data.get("@type")
    # Check per type
    if t == "Organization" and not data.get("name"):
        return False, "Organization missing name"
    if t == "Article" and not data.get("headline"):
        return False, "Article missing headline"
    if t == "FAQPage" and not data.get("mainEntity"):
        return False, "FAQPage missing mainEntity"
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
    Generate schema.org JSON-LD

    biz_type: one of Organization | LocalBusiness | Article | FAQPage | OfferCatalog | Product
    site_name: default name of the site/account
    site_url: absolute site URL
    language: optional site language
    extras: additional payload fields (faq list, author, logo, sameAs etc.)
    rag_context: optional string with context text from site (future RAG integration)
    """
    extras = extras or {}
    bt = (biz_type or "Organization").strip()
    name = site_name or urlparse(site_url).netloc

    payload = {
        "biz_type": bt,
        "defaults": {"name": name, "url": site_url, "language": language},
        "extras": extras,
        "context": rag_context or "(no extra site context)"
    }

    prompt = f"Generate a JSON-LD object for:\n{json.dumps(payload, ensure_ascii=False)}"

    data = _call_llm(prompt)

    # Fallback if model fails
    if not isinstance(data, dict) or not data.get("@type"):
        return _fallback_schema(bt, name, site_url)

    # Ensure context always present
    data.setdefault("@context", "https://schema.org")

    # Validate
    ok, err = validate_schema(data, bt)
    if not ok:
        # If invalid → fallback minimal schema
        return _fallback_schema(bt, name, site_url)

    return data
