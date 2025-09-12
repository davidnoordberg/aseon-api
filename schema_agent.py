# schema_agent.py
# Aseon - Schema.org JSON-LD generator (multi-type, payload controls, strict validation)

import os
import json
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

_BASE_SYS = """
You are an expert Schema.org JSON-LD generator.
Return a SINGLE valid JSON object. No comments, no markdown, no backticks.
Rules:
- Always include "@context":"https://schema.org".
- Output must be strictly valid JSON-LD for the requested type.
- Never invent data (no fake phone, address, price). Omit unknown optional fields.
- Keep FAQ answers ≤ 80 words, factual, no marketing fluff.
- Follow the provided language when writing text fields if present.
Supported types & minimum fields:
1) Organization / LocalBusiness:
   - @type, name, url
   - Optional: logo, sameAs[], address, telephone
2) Article:
   - @type, headline, description, author {name, url?}, datePublished (ISO8601), mainEntityOfPage (url), image (url)
3) FAQPage:
   - @type, mainEntity: [ { @type:"Question", name, acceptedAnswer: { @type:"Answer", text } } ]
4) OfferCatalog:
   - @type, name, description, url, itemListElement: [ { @type:"ListItem", position, item: { @type:"Offer", name, url } } ]
5) Product:
   - @type, name, description, url
General:
- Use fields we provide as hints (defaults/extras). Do not copy the context text verbatim; synthesize concise, factual fields.
- If the requested FAQ count is provided, produce AT MOST that many questions.
"""

def _clamp_words(text: str, max_words: int = 80) -> str:
    if not isinstance(text, str):
        return ""
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()

def _fallback_schema(biz_type: str, site_name: str, site_url: str) -> Dict[str, Any]:
    bt = biz_type or "Organization"
    base: Dict[str, Any] = {
        "@context": "https://schema.org",
        "@type": bt,
        "name": site_name,
        "url": site_url
    }
    if bt == "FAQPage":
        base["mainEntity"] = [{
            "@type": "Question",
            "name": "What is this site about?",
            "acceptedAnswer": {"@type": "Answer", "text": "This site provides information about our products and services."}
        }]
    return base

def _ensure_context_type(data: Dict[str, Any], biz_type: str) -> None:
    data.setdefault("@context", "https://schema.org")
    data["@type"] = data.get("@type") or biz_type or "Organization"

def _merge_sameas(data: Dict[str, Any], same_as: Optional[List[str]]) -> None:
    if not same_as:
        return
    cur = data.get("sameAs") or []
    if not isinstance(cur, list):
        cur = []
    merged = []
    seen = set()
    for url in list(cur) + list(same_as):
        if not isinstance(url, str):
            continue
        u = url.strip()
        if not u or u in seen:
            continue
        if not (u.startswith("http://") or u.startswith("https://")):
            continue
        seen.add(u)
        merged.append(u)
    if merged:
        data["sameAs"] = merged

def _trim_faq_answers(data: Dict[str, Any], max_q: Optional[int], lang: Optional[str]) -> None:
    if data.get("@type") != "FAQPage":
        return
    ents = data.get("mainEntity")
    if not isinstance(ents, list):
        data["mainEntity"] = []
        return
    out = []
    for i, q in enumerate(ents):
        if not isinstance(q, dict):
            continue
        name = q.get("name")
        ans = (q.get("acceptedAnswer") or {})
        text = ans.get("text") if isinstance(ans, dict) else None
        if not name or not text:
            continue
        ans["@type"] = "Answer"
        ans["text"] = _clamp_words(str(text), 80)
        out.append({
            "@type": "Question",
            "name": str(name).strip(),
            "acceptedAnswer": ans
        })
        if max_q is not None and len(out) >= max_q:
            break
    data["mainEntity"] = out

def _validate_minimal(data: Dict[str, Any], biz_type: str) -> (bool, Optional[str]):
    if not isinstance(data, dict):
        return False, "Schema is not an object"
    if "@type" not in data:
        return False, "Missing @type"
    t = data.get("@type")
    # Normalize common alias: LocalBusiness subtype is allowed
    if biz_type in ("Organization", "LocalBusiness") and t in ("Organization", "LocalBusiness"):
        if not data.get("name") or not data.get("url"):
            return False, "Organization/LocalBusiness missing name or url"
        return True, None
    if biz_type == "Article" or t == "Article":
        required = ["headline", "description", "author", "datePublished", "mainEntityOfPage", "image"]
        missing = [k for k in required if not data.get(k)]
        if missing:
            return False, "Article missing: " + ",".join(missing)
        return True, None
    if biz_type == "FAQPage" or t == "FAQPage":
        ents = data.get("mainEntity")
        if not isinstance(ents, list) or not ents:
            return False, "FAQPage missing mainEntity"
        for q in ents:
            if not isinstance(q, dict): return False, "FAQPage invalid Question"
            if q.get("@type") != "Question": return False, "FAQPage Question missing @type"
            a = q.get("acceptedAnswer")
            if not isinstance(a, dict) or a.get("@type") != "Answer" or not a.get("text"):
                return False, "FAQPage acceptedAnswer invalid"
        return True, None
    if biz_type == "OfferCatalog" or t == "OfferCatalog":
        if not data.get("name") or not data.get("description") or not data.get("url"):
            return False, "OfferCatalog missing basic fields"
        il = data.get("itemListElement")
        if not isinstance(il, list) or not il:
            return False, "OfferCatalog missing itemListElement"
        return True, None
    if biz_type == "Product" or t == "Product":
        if not data.get("name") or not data.get("description") or not data.get("url"):
            return False, "Product missing name/description/url"
        return True, None
    # Fallback: at least have name+url if present
    return True, None

def _call_llm(prompt: str, use_json_mode: bool = True) -> Optional[Dict[str, Any]]:
    try:
        if use_json_mode:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": _BASE_SYS},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
        else:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": _BASE_SYS},
                          {"role": "user", "content": prompt}],
                temperature=0.3,
            )
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception as e1:
        # Retry without JSON mode once
        if use_json_mode:
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": _BASE_SYS},
                              {"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                content = resp.choices[0].message.content
                return json.loads(content)
            except Exception as e2:
                print(json.dumps({"level": "ERROR", "msg": "schema_llm_failed", "error": str(e2)}), flush=True)
        else:
            print(json.dumps({"level": "ERROR", "msg": "schema_llm_failed", "error": str(e1)}), flush=True)
        return None

def generate_schema(
    biz_type: str,
    site_name: Optional[str],
    site_url: str,
    language: Optional[str] = None,
    country: Optional[str] = None,
    extras: Optional[Dict[str, Any]] = None,
    rag_context: Optional[str] = None,
    faq_count: Optional[int] = None
) -> Dict[str, Any]:
    """
    Returns ONLY the JSON-LD object (dict). Caller (general_agent) can wrap it with metadata.
    """
    extras = extras or {}
    bt = (biz_type or "Organization").strip()
    name = site_name or urlparse(site_url).netloc

    # Build payload for the prompt
    request_payload = {
        "biz_type": bt,
        "defaults": {
            "name": name,
            "url": site_url,
            "language": language,
            "country": country
        },
        "controls": {
            "faq_max_items": int(faq_count) if (isinstance(faq_count, int) and faq_count > 0) else None
        },
        "extras": {
            # sameAs can be provided here; we will merge after generation as well
            "sameAs": list(extras.get("sameAs", [])) if isinstance(extras.get("sameAs"), list) else []
        },
        "context_excerpt": (rag_context or "")[:2000]  # keep prompt compact
    }

    prompt = (
        "Generate a single JSON-LD object for the following request. "
        "Respect the rules above (valid JSON, required fields, ≤80 words for FAQ answers, no invented data). "
        "Do not include explanations.\n\n"
        + json.dumps(request_payload, ensure_ascii=False)
    )

    data = _call_llm(prompt, use_json_mode=True)
    if not isinstance(data, dict) or not data:
        data = _fallback_schema(bt, name, site_url)

    # Normalize type/context
    _ensure_context_type(data, bt)

    # Merge sameAs from extras (dedupe)
    _merge_sameas(data, request_payload["extras"]["sameAs"])

    # Enforce FAQ trims & count
    _trim_faq_answers(data, request_payload["controls"]["faq_max_items"], language)

    # Validate
    ok, err = _validate_minimal(data, bt)
    if not ok:
        print(json.dumps({"level": "WARN", "msg": "schema_invalid", "error": err}), flush=True)
        data = _fallback_schema(bt, name, site_url)
        # If fallback is FAQ, still clamp/trim to count
        _trim_faq_answers(data, request_payload["controls"]["faq_max_items"], language)

    return data
