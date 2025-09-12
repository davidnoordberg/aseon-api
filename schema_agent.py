# schema_agent.py
import os, json
from urllib.parse import urlparse
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

_BASE_SYS = """
You are an expert Schema.org JSON-LD generator.
Rules:
- Return a single valid JSON object, no comments/explanations.
- Always include "@context":"https://schema.org".
- Conform strictly to Schema.org.
Types:
1) Organization/LocalBusiness: name, url, logo?, sameAs[], address?, telephone?
2) Article: headline, description?, author{name,url}, datePublished?, mainEntityOfPage?, image?
3) FAQPage: mainEntity:[{ @type:Question, name, acceptedAnswer{ @type:Answer, text } }]
   - Answers ≤ 80 words, factual.
4) OfferCatalog/Product: name, description, url, itemListElement[].
"""

def _call_llm(prompt: str) -> dict | None:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content":_BASE_SYS},{"role":"user","content":prompt}],
            response_format={"type":"json_object"},
            temperature=0.3,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role":"system","content":_BASE_SYS},{"role":"user","content":prompt}],
                temperature=0.3,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e2:
            print(json.dumps({"level":"ERROR","msg":"schema_llm_failed","error":str(e2)}), flush=True)
            return None

def _fallback_schema(biz_type: str, site_name: str, site_url: str) -> dict:
    return {"@context":"https://schema.org","@type":biz_type,"name":site_name,"url":site_url}

def validate_schema(data: dict, biz_type: str) -> tuple[bool, str | None]:
    if not isinstance(data, dict): return False, "Schema is not a dict"
    t = data.get("@type")
    if not t: return False, "Missing @type"
    if t == "FAQPage" and not data.get("mainEntity"): return False, "FAQPage missing mainEntity"
    if t == "Article" and not data.get("headline"): return False, "Article missing headline"
    if t in ("Organization","LocalBusiness") and not data.get("name"): return False, "Organization missing name"
    return True, None

def generate_schema(biz_type: str, site_name: str | None, site_url: str,
                    language: str | None = None, extras: dict | None = None,
                    rag_context: str | None = None) -> dict:
    extras = extras or {}
    bt = (biz_type or "Organization").strip()
    name = site_name or urlparse(site_url).netloc
    count = int(extras.get("count", 3))

    # ---- Direct build for FAQPage if we already have FAQs ----
    if bt == "FAQPage" and isinstance(extras.get("faqs"), list) and extras["faqs"]:
        faqs = extras["faqs"][:count]
        main = []
        for f in faqs:
            q = (f.get("q") or "").strip()
            a = (f.get("a") or "").strip()
            if not q or not a:
                continue
            main.append({
                "@type": "Question",
                "name": q,
                "acceptedAnswer": {"@type": "Answer", "text": a}
            })
        if main:
            return {"@context":"https://schema.org","@type":"FAQPage","mainEntity": main}
        # if somehow empty, fall through to fallback

    # ---- LLM path for other types / fallback ----
    payload = {
        "biz_type": bt,
        "defaults": {"name": name, "url": site_url, "language": language},
        "extras": extras,
        "context": rag_context or "(no extra site context)"
    }
    prompt = f"Generate a JSON-LD object for:\n{json.dumps(payload, ensure_ascii=False)}"
    data = _call_llm(prompt) or _fallback_schema(bt, name, site_url)

    # Small deterministic tweak for Article: prefer extras.url as mainEntityOfPage
    if bt == "Article" and extras.get("url"):
        data.setdefault("mainEntityOfPage", extras["url"])
    data.setdefault("@context", "https://schema.org")
    data.setdefault("@type", bt)

    ok, err = validate_schema(data, bt)
    if not ok:
        print(json.dumps({"level":"WARN","msg":"schema_invalid","error":err}), flush=True)
        data = _fallback_schema(bt, name, site_url)
    return data
