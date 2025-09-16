import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

from openai import OpenAI
from psycopg.rows import dict_row

# -----------------------------
# Config
# -----------------------------
OPENAI_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "30"))
_openai_key = os.getenv("OPENAI_API_KEY")
_openai_client = OpenAI(api_key=_openai_key) if _openai_key else None

# Product-wide toggles (kan je later desgewenst via payload overschrijven)
DEFAULT_TOGGLES = {
    "language_mode": "auto_default_en",            # detecteer, anders EN
    "faq_mode": "strict",                          # FAQ alleen op type=faq
    "micro_faq_enabled": False,                    # nooit micro-FAQ op non-FAQ
    "max_qas_faq": 6,                              # 3–6 Q&A op FAQ
    "emit_jsonld_when_visible_only": True,         # JSON-LD alleen als content zichtbaar is
}


# -----------------------------
# Helpers (DB)
# -----------------------------
def _fetch_latest_job(conn, site_id: str, jtype: str):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT output
              FROM jobs
             WHERE site_id=%s AND type=%s AND status='done'
             ORDER BY finished_at DESC NULLS LAST, created_at DESC
             LIMIT 1
            """,
            (site_id, jtype),
        )
        r = cur.fetchone()
        return (r or {}).get("output") if r else None


def _fetch_site_meta(conn, site_id: str) -> Dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT s.url, s.language, s.country, a.name AS account_name
              FROM sites s
              JOIN accounts a ON a.id = s.account_id
             WHERE s.id=%s
            """,
            (site_id,),
        )
        return cur.fetchone() or {}


# -----------------------------
# URL / normalize
# -----------------------------
def _norm_url(u: str) -> str:
    if not u:
        return ""
    p = urlsplit(u)
    scheme = p.scheme or "https"
    host = p.hostname or ""
    path = (p.path or "/")
    if not path:
        path = "/"
    if not path.startswith("/"):
        path = "/" + path
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return urlunsplit((scheme, host, path, "", ""))


# -----------------------------
# Language
# -----------------------------
def _autolang(site_meta: Dict[str, Any], crawl: Optional[Dict[str, Any]]) -> str:
    lang = (site_meta.get("language") or "").strip().lower()
    if lang:
        return "nl" if lang.startswith("nl") else "en"
    c = (site_meta.get("country") or "").strip().lower()
    if c in ("nl", "nld", "netherlands"):
        return "nl"
    return "en"


def _page_language(page: Dict[str, Any], default_lang: str) -> str:
    return (page.get("language") or default_lang).split("-")[0].lower()


# -----------------------------
# Page-type classifier (rules-first, generiek)
# -----------------------------
def _classify_page_type(url: str, title: str, h1: str) -> str:
    u = url.lower()
    path = urlsplit(u).path or "/"
    t = (title or "").lower()
    h = (h1 or "").lower()

    def has(*keys: str) -> bool:
        s = " " + " ".join(keys) + " "
        return any(k in path or k in t or k in h for k in keys)

    if path == "/" or path in ("/index", "/index.html"):
        return "home"
    if has("/faq", " faq"):
        return "faq"
    if has("/contact", " contact"):
        return "contact"
    if has("/about", "/about-us", "/over-ons", " about ", " over ons"):
        return "about"
    if has("/pricing", "/prices", "/prijs", "/subscription", " pricing", " price ", " subscription"):
        return "pricing"
    if has("/privacy", "/terms", "/legal", " privacy", " terms", " voorwaarden"):
        return "legal"
    if has("/blog/") and not path.endswith("/blog/"):
        return "blog_post"
    if has("/blog", "/nieuws", "/news", "/insights", "/resources"):
        return "listing"
    if has("/services", "/service", "/solutions", " service", " solution"):
        return "service"
    return "other"


# -----------------------------
# Q&A helpers (FAQ only)
# -----------------------------
def _word_count(text: str) -> int:
    return len((text or "").strip().split())


def _has_faq_schema(page: Dict[str, Any]) -> bool:
    types = [t.lower() for t in (page.get("jsonld_types") or [])]
    return any(t == "faqpage" for t in types)


def _shorten(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[: n - 1] + "…")


def _llm_qas_from_page(lang: str, title: str, h1: str, body_preview: str, kb_ctx: str, n: int = 4) -> List[Dict[str, str]]:
    # Fallback zonder OpenAI-key
    if not _openai_client:
        topic = (h1 or title or "").strip() or ("deze pagina" if lang == "nl" else "this page")
        if lang == "nl":
            qs = [
                {"q": f"Wat is {topic.lower()}?", "a": f"{topic} kort uitgelegd in eenvoudige taal (≤80 woorden)."},
                {"q": f"Hoe werkt {topic.lower()}?", "a": "Korte uitleg in 40–60 woorden met 2–3 concrete stappen."},
                {"q": f"Wat levert {topic.lower()} op?", "a": "Resultaten/voordelen in 40–60 woorden met 1 meetbaar voorbeeld."},
                {"q": f"Hoe begin ik?", "a": "3–5 genummerde mini-stappen (≤70 woorden)."},
            ]
        else:
            qs = [
                {"q": f"What is {topic}?", "a": f"{topic} explained plainly in ≤80 words."},
                {"q": "How does it work?", "a": "40–60 words with 2–3 concrete steps."},
                {"q": "What are the benefits?", "a": "40–60 words with one measurable example."},
                {"q": "How do I get started?", "a": "3–5 numbered mini-steps (≤70 words)."},
            ]
        return qs[:n]

    sys = (
        "You create snippet-ready Q&A for Answer/Generative Engines. "
        "Write in the requested language. Each answer must be ≤80 words, lead with the answer, "
        "use only facts from the provided context."
    )
    user = {
        "language": lang,
        "title": title,
        "h1": h1,
        "body_preview": _shorten(body_preview, 1200),
        "kb": _shorten(kb_ctx, 1500),
        "n": n,
    }
    try:
        resp = _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            timeout=OPENAI_TIMEOUT_SEC,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
        )
        txt = resp.choices[0].message.content.strip()
        m = re.search(r"\[.*\]", txt, re.S)
        data = json.loads(m.group(0)) if m else []
        out = []
        for x in data:
            q = (x.get("q") or x.get("question") or "").strip()
            a = (x.get("a") or x.get("answer") or "").strip()
            if q and a:
                out.append({"q": q, "a": a})
        return out[:n]
    except Exception:
        return []


def _faq_html_block(qas: List[Dict[str, str]], lang: str) -> str:
    label = "Veelgestelde vragen" if lang == "nl" else "Frequently asked questions"
    lis = []
    for qa in qas:
        q = qa["q"].strip()
        a = qa["a"].strip()
        lis.append(
            f"""<li class="faq-item">
  <h3 class="faq-q">{q}</h3>
  <p class="faq-a">{a}</p>
</li>"""
        )
    return f"""<section id="faq" aria-labelledby="faq-title">
  <h2 id="faq-title">{label}</h2>
  <ul class="faq-list">
    {''.join(lis)}
  </ul>
</section>"""


def _faq_jsonld(qas: List[Dict[str, str]]) -> Dict[str, Any]:
    items = []
    for qa in qas:
        items.append(
            {
                "@type": "Question",
                "name": qa["q"],
                "acceptedAnswer": {"@type": "Answer", "text": qa["a"]},
            }
        )
    return {"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": items}


# -----------------------------
# Copy recepten (niet-FAQ)
# -----------------------------
def _llm_copy_recipe(lang: str, page_type: str, title: str, h1: str, body_preview: str) -> Dict[str, Any]:
    # Fallback (zonder LLM) – generieke, nette blokken
    def fb_text(en: str, nl: str) -> str:
        return nl if lang == "nl" else en

    base_h1 = (h1 or title or ("Welkom" if lang == "nl" else "Welcome")).strip()
    out: Dict[str, Any] = {
        "hero": {
            "h1": base_h1[:80],
            "subhead": fb_text("A concise value promise in one sentence.", "Eén zin met de kernbelofte."),
            "primary_cta": fb_text("Get started", "Aan de slag"),
            "secondary_cta": fb_text("Learn more", "Meer weten"),
        },
        "value_props": [
            {"title": fb_text("Fast", "Snel"), "desc": fb_text("Quick outcomes you can measure.", "Snelle, meetbare resultaten.")},
            {"title": fb_text("Clear", "Duidelijk"), "desc": fb_text("No fluff; just what works.", "Geen ruis; alleen wat werkt.")},
            {"title": fb_text("Proven", "Bewezen"), "desc": fb_text("Backed by real examples.", "Gestaafd met voorbeelden.")},
        ],
        "steps": [
            fb_text("1) Assess needs", "1) Behoefte bepalen"),
            fb_text("2) Plan actions", "2) Actieplan maken"),
            fb_text("3) Implement & review", "3) Implementeren & evalueren"),
        ],
        "proof": [fb_text("Used by teams like yours.", "Gebruikt door teams zoals dat van jou.")],
        "ctas": [
            {"label": fb_text("Contact us", "Neem contact op"), "href_hint": "/contact"},
            {"label": fb_text("See pricing", "Bekijk prijzen"), "href_hint": "/pricing"},
        ],
    }

    if not _openai_client:
        return out

    sys = (
        "You produce compact, snippet-ready page blocks for the given page type. "
        "Return JSON with keys: hero{h1,subhead,primary_cta,secondary_cta}, "
        "value_props[{title,desc}×3-4], steps[3-5], proof[1-3], ctas[{label,href_hint}×1-3]. "
        "Language matches the requested language; keep each block concise."
    )
    user = json.dumps(
        {
            "language": lang,
            "page_type": page_type,
            "title": title,
            "h1": h1,
            "body_preview": _shorten(body_preview, 1200),
        },
        ensure_ascii=False,
    )
    try:
        resp = _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            timeout=OPENAI_TIMEOUT_SEC,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        )
        data = json.loads(resp.choices[0].message.content)
        # Sanity + fallbacks
        return {
            "hero": {
                "h1": (data.get("hero", {}) or {}).get("h1") or out["hero"]["h1"],
                "subhead": (data.get("hero", {}) or {}).get("subhead") or out["hero"]["subhead"],
                "primary_cta": (data.get("hero", {}) or {}).get("primary_cta") or out["hero"]["primary_cta"],
                "secondary_cta": (data.get("hero", {}) or {}).get("secondary_cta") or out["hero"]["secondary_cta"],
            },
            "value_props": data.get("value_props") or out["value_props"],
            "steps": data.get("steps") or out["steps"],
            "proof": data.get("proof") or out["proof"],
            "ctas": data.get("ctas") or out["ctas"],
        }
    except Exception:
        return out


# -----------------------------
# AEO scoring (type-aware)
# -----------------------------
_EXPECTED_SCHEMA = {
    "home": {"website", "organization"},
    "about": {"aboutpage", "organization", "person"},
    "service": {"service"},
    "pricing": {"product", "offer"},
    "faq": {"faqpage"},
    "contact": {"contactpage"},
    "legal": set(),  # meestal niets extra’s nodig
    "blog_post": {"article", "blogposting"},
    "listing": {"collectionpage", "blog"},
    "other": set(),
}


def _has_expected_schema(jsonld_types: List[str], page_type: str) -> bool:
    types = {t.lower() for t in (jsonld_types or [])}
    expected = _EXPECTED_SCHEMA.get(page_type, set())
    return any(t in types for t in expected)


def _score_nonfaq_page(
    page: Dict[str, Any],
    page_type: str,
    intro_words: int,
    has_expected: bool,
    canonical_ok: bool,
    title_ok: bool,
    meta_ok: bool,
    has_cta_link: bool,
) -> Tuple[int, List[str], Dict[str, int]]:
    score = 0
    issues: List[str] = []
    metrics = {
        "intro_20_80w": 1 if 20 <= intro_words <= 80 else 0,
        "has_expected_schema": 1 if has_expected else 0,
        "title_in_range": 1 if title_ok else 0,
        "meta_in_range": 1 if meta_ok else 0,
        "canonical_ok": 1 if canonical_ok else 0,
        "cta_present": 1 if has_cta_link else 0,
    }

    # Weights (total max 45 on non-FAQ signals)
    score += metrics["intro_20_80w"] * 10
   
