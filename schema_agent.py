# schema_agent.py
import os, json
from urllib.parse import urlparse
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Vast instructieblok
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
2. Article
   - Fields: headline, description, author{name,url}, datePublished, mainEntityOfPage, image
3. FAQPage
   - Fields: mainEntity: [Question + acceptedAnswer{text}]
   - Answers must be ≤80 words, factual, no marketing fluff
4. OfferCatalog / Product
   - Fields: name, description, url, itemListElement[]
"""

def _call_llm(prompt: str) -> dict:
    # Eerste poging met strikt JSON-format
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
    except Exception as e1:
        # Tweede poging zonder response_format (meer vrijheid)
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
        except Exception as e2:
            print(json.dumps({
                "level": "ERROR",
                "msg": "schema_llm_failed",
                "error": str(e2)
            }), flush=True)
            return {}

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

    # Fallback als AI faalt
    if not isinstance(data, dict) or not data.get("@type"):
        data = _fallback_schema(bt, name, site_url)

    # Context altijd verplicht
    data.setdefault("@context", "https://schema.org")

    # Validatie
    ok, err = validate_schema(data, bt)
    if not ok:
        print(json.dumps({
            "level": "WARN",
            "msg": "schema_invalid",
            "error": err
        }), flush=True)
        data = _fallback_schema(bt, name, site_url)

    return data
