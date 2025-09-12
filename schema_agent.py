# schema_agent.py
import os, json
from urllib.parse import urlparse
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

_BASE_SYS = """
You are an expert Schema.org JSON-LD generator.
Return ONE valid JSON object only. No explanations. No comments.
ALWAYS include "@context": "https://schema.org".
NEVER invent facts not present in the provided site context or explicit extras.
If data is missing, omit the field instead of guessing.

Supported types:
1) Organization / LocalBusiness
   Fields: name, url, logo?, sameAs[], address?, telephone?
2) Article
   Fields: headline, description, author{name,url}, datePublished, mainEntityOfPage, image
3) FAQPage
   Fields: mainEntity: [ { @type: "Question", name, acceptedAnswer: { @type: "Answer", text } } ]
   - Q&As MUST be grounded in provided `context` or `extras`. No hallucinations.
   - Each Answer MUST be ≤ 80 words, factual, neutral.
4) OfferCatalog / Product
   Fields: name, description, url, itemListElement[]

Rules:
- Follow provided language if available.
- Use extras.sameAs exactly as given. Do not add random profiles.
- No phone/address/prices unless provided.
"""

def _call_llm(prompt: str, expect_json: bool = True) -> dict | None:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content":_BASE_SYS},{"role":"user","content":prompt}],
            response_format={"type":"json_object"} if expect_json else None,
            temperature=0.2,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role":"system","content":_BASE_SYS},{"role":"user","content":prompt}],
                temperature=0.2,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e2:
            print(json.dumps({"level":"ERROR","msg":"schema_llm_failed","error":str(e2)}), flush=True)
            return None

def _fallback_schema(biz_type: str, site_name: str, site_url: str) -> dict:
    return {"@context":"https://schema.org","@type":biz_type,"name":site_name,"url":site_url}

def validate_schema(data: dict, biz_type: str) -> tuple[bool, str | None]:
    if not isinstance(data, dict): return False, "Schema is not a dict"
    if "@type" not in data: return False, "Missing @type"
    t = data.get("@type")
    if t == "Organization" and not data.get("name"): return False, "Organization missing name"
    if t == "Article" and not data.get("headline"): return False, "Article missing headline"
    if t == "FAQPage":
        main = data.get("mainEntity")
        if not main or not isinstance(main, list) or len(main) == 0:
            return False, "FAQPage missing mainEntity"
        for q in main:
            if not isinstance(q, dict): return False, "FAQ item is not object"
            if q.get("@type") != "Question": return False, "FAQ item must be Question"
            a = (q.get("acceptedAnswer") or {})
            if a.get("@type") != "Answer" or not a.get("text"):
                return False, "FAQ item missing acceptedAnswer.text"
    return True, None

def _trim_faq_answers(data: dict, max_words: int):
    main = data.get("mainEntity") or []
    cleaned = []
    for q in main:
        if not isinstance(q, dict): continue
        a = (q.get("acceptedAnswer") or {}).get("text") or ""
        words = " ".join(a.split()).split()
        if len(words) > max_words: a = " ".join(words[:max_words])
        cleaned.append({"@type":"Question","name":q.get("name"),
                        "acceptedAnswer":{"@type":"Answer","text":a}})
    data["mainEntity"] = cleaned

def generate_schema(
    biz_type: str,
    site_name: str | None,
    site_url: str,
    language: str | None = None,
    extras: dict | None = None,
    rag_context: str | None = None,
    faq_count: int = 3,
    max_faq_words: int = 80
) -> dict:
    extras = extras or {}
    bt = (biz_type or "Organization").strip()
    name = site_name or urlparse(site_url).netloc

    payload = {
        "biz_type": bt,
        "defaults": {"name": name, "url": site_url, "language": language},
        "extras": extras,
        "faq_constraints": {"count": faq_count, "max_words": max_faq_words},
        "context": rag_context or "(no context provided)"
    }

    prompt = (
        "Generate strictly grounded Schema.org JSON-LD.\n"
        "Only include facts present in `context` or `extras`.\n"
        "If a field is unknown, omit it. Never invent services or contact data.\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )

    data = _call_llm(prompt, expect_json=True)
    if not isinstance(data, dict) or not data:
        data = _fallback_schema(bt, name, site_url)

    data.setdefault("@context", "https://schema.org")
    data.setdefault("@type", bt)

    if data.get("@type") == "FAQPage":
        data["mainEntity"] = (data.get("mainEntity") or [])[:faq_count]
        _trim_faq_answers(data, max_faq_words)

    ok, err = validate_schema(data, bt)
    if not ok:
        print(json.dumps({"level":"WARN","msg":"schema_invalid","error":err}), flush=True)
        data = _fallback_schema(bt, name, site_url)

    return data
