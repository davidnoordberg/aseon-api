# aeo_agent.py

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

DEFAULT_TOGGLES = {
    "language_mode": "auto_default_en",
    "faq_mode": "strict",                 # FAQ alleen als het echt FAQ is
    "micro_faq_enabled": False,
    "max_qas_faq": 6,                     # toon/patch max 6
    "emit_jsonld_when_visible_only": True,
}

# -----------------------------
# DB helpers
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
# URL / language
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
# Page-type classifier
# -----------------------------
def _classify_page_type(url: str, title: str, h1: str) -> str:
    u = url.lower()
    path = urlsplit(u).path or "/"
    t = (title or "").lower()
    h = (h1 or "").lower()

    def has(*keys: str) -> bool:
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
    if "/blog/" in path and not path.endswith("/blog/"):
        return "blog_post"
    if has("/blog", "/nieuws", "/news", "/insights", "/resources"):
        return "listing"
    if has("/services", "/service", "/solutions", " service", " solution"):
        return "service"
    return "other"

# -----------------------------
# Small utils
# -----------------------------
def _shorten(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[: n - 1] + "…")

def _word_count(text: str) -> int:
    return len((text or "").strip().split())

def _first_paragraph(paragraphs: List[str]) -> str:
    for p in (paragraphs or []):
        if _word_count(p) >= 10:
            return p.strip()
    return (paragraphs or [""])[0].strip() if (paragraphs or []) else ""

# -----------------------------
# JSON-LD helpers
# -----------------------------
def _jsonld_objects(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for k in ("jsonld", "json_ld", "structured_data", "ld_json"):
        v = page.get(k)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    out.append(item)
        elif isinstance(v, dict):
            out.append(v)
    return out

def _type_lower(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(t).lower() for t in x]
    if isinstance(x, str):
        return [x.lower()]
    return []

def _has_faq_schema(page: Dict[str, Any]) -> bool:
    # via simple type list
    types = [t.lower() for t in (page.get("jsonld_types") or [])]
    if any(t == "faqpage" for t in types):
        return True
    # via full JSON-LD objects
    for obj in _jsonld_objects(page):
        for t in _type_lower(obj.get("@type")):
            if t == "faqpage":
                return True
    return False

# -----------------------------
# Q&A extraction
# -----------------------------
_Q_HINT_EN = re.compile(r"\?$|^(what|how|why|when|where|who|can|does|is|are)\b", re.I)
_Q_HINT_NL = re.compile(r"\?$|^(wat|hoe|waarom|wanneer|waar|wie|kan|doet|is|zijn)\b", re.I)

def _extract_qas_from_structured(page: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Lees Q&A uit allerlei bekende velden (breed gedekt) en uit JSON-LD FAQPage.
    """
    out: List[Dict[str, str]] = []

    # 1) Polulaire veldnamen (arrays van {q,a} of {question,answer})
    candidates = []
    for key in ("faqs", "faq_items", "faq_qas", "faq", "faq_list", "qa_pairs", "q_and_a", "qna", "questions"):
        v = page.get(key)
        if isinstance(v, list):
            candidates.extend(v)
        elif isinstance(v, dict) and isinstance(v.get("items"), list):
            candidates.extend(v.get("items") or [])

    for it in candidates:
        if not isinstance(it, dict):
            continue
        q = (it.get("q") or it.get("question") or it.get("name") or "").strip()
        a = (it.get("a") or it.get("answer") or (it.get("acceptedAnswer") or {}).get("text") or "").strip()
        if q and a:
            out.append({"q": q, "a": a})

    # 2) JSON-LD FAQPage → mainEntity
    for obj in _jsonld_objects(page):
        types = _type_lower(obj.get("@type"))
        if "faqpage" in types:
            main = obj.get("mainEntity") or []
            if isinstance(main, list):
                for m in main:
                    if not isinstance(m, dict):
                        continue
                    q = (m.get("name") or m.get("question") or "").strip()
                    acc = m.get("acceptedAnswer") or {}
                    a = (acc.get("text") or acc.get("answer") or "").strip() if isinstance(acc, dict) else ""
                    if q and a:
                        out.append({"q": q, "a": a})

    # dedupe op vraag
    dedup, seen = [], set()
    for qa in out:
        k = qa["q"].strip().lower()
        if k in seen:
            continue
        seen.add(k)
        dedup.append(qa)
    return dedup

def _extract_qas_from_headings_and_paragraphs(page: Dict[str, Any], lang: str) -> List[Dict[str, str]]:
    """
    Heuristiek: neem h2/h3/h4 die eruitzien als vragen en koppel die aan de dichtstbijzijnde
    VOLGENDE paragraaf met ≥12 woorden.
    """
    headings = (page.get("h2") or []) + (page.get("h3") or []) + (page.get("h4") or [])
    paragraphs = page.get("paragraphs") or []
    qre = _Q_HINT_NL if lang == "nl" else _Q_HINT_EN

    # indexeer eerste zinnige paragrafen
    para_ix = [i for i, p in enumerate(paragraphs) if _word_count(p) >= 12]

    qas: List[Dict[str, str]] = []
    for q in headings:
        if not q:
            continue
        q = q.strip()
        if _word_count(q) > 25 or not qre.search(q):
            continue
        # vind dichtstbijzijnde volgende paragraaf
        answer = ""
        if para_ix:
            # neem de eerste para-index — in echte DOM zou je posities hebben; hier nemen we de eerstvolgende
            answer = paragraphs[para_ix[0]].strip()
            # schuif het venster op
            para_ix = para_ix[1:]
        if q and answer:
            qas.append({"q": q, "a": answer})

    # dedupe
    dedup, seen = [], set()
    for qa in qas:
        k = qa["q"].strip().lower()
        if k in seen:
            continue
        seen.add(k)
        dedup.append(qa)
    return dedup

def _trim_to_80w(q: str, a: str) -> Tuple[str, str, bool]:
    words = a.split()
    if len(words) <= 80:
        return q, a, False
    return q, " ".join(words[:80]) + "…", True

# -----------------------------
# LLM helpers
# -----------------------------
def _llm_qas_from_page(lang: str, title: str, h1: str, body_preview: str, kb_ctx: str, n: int = 4) -> List[Dict[str, str]]:
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
        "and only use facts from the provided context. Return a JSON array of {q,a}."
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

# -----------------------------
# FAQ HTML/JSON-LD builders
# -----------------------------
def _faq_html_block(qas: List[Dict[str, str]], lang: str) -> str:
    label = "Veelgestelde vragen" if lang == "nl" else "Frequently asked questions"
    items = []
    for qa in qas:
        q = qa["q"].strip()
        a = qa["a"].strip()
        items.append(
f"""<li class="faq-item">
  <h3 class="faq-q">{q}</h3>
  <p class="faq-a">{a}</p>
</li>"""
        )
    return f"""<section id="faq" aria-labelledby="faq-title">
  <h2 id="faq-title">{label}</h2>
  <ul class="faq-list">
    {''.join(items)}
  </ul>
</section>"""

def _faq_jsonld(qas: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {"@type": "Question", "name": qa["q"], "acceptedAnswer": {"@type": "Answer", "text": qa["a"]}}
            for qa in qas
        ],
    }

# -----------------------------
# Non-FAQ content blocks
# -----------------------------
def _llm_copy_recipe(lang: str, page_type: str, title: str, h1: str, body_preview: str) -> Dict[str, Any]:
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
        {"language": lang, "page_type": page_type, "title": title, "h1": h1, "body_preview": _shorten(body_preview, 1200)},
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

def _patch_from_blocks(url: str, blocks: Dict[str, Any], lang: str) -> List[Dict[str, Any]]:
    patches: List[Dict[str, Any]] = []
    hero = blocks.get("hero") or {}
    if hero:
        html = f"""<section class="hero">
  <h1>{hero.get('h1','')}</h1>
  <p class="subhead">{hero.get('subhead','')}</p>
  <div class="ctas">
    <a href="#" class="btn btn-primary">{hero.get('primary_cta','')}</a>
    <a href="#" class="btn btn-secondary">{hero.get('secondary_cta','')}</a>
  </div>
</section>"""
        patches.append({
            "url": url, "field": "hero",
            "problem": "Missing/weak hero section" if lang == "en" else "Ontbrekende/zwakke hero-sectie",
            "current": "(none)", "proposed": None, "html_patch": html,
            "category": "body", "severity": 2, "impact": 5, "effort": 2, "priority": 5.5, "patchable": True
        })
    if hero.get("subhead"):
        patches.append({
            "url": url, "field": "subhead",
            "problem": "Clarify value in one sentence" if lang == "en" else "Verduidelijk de waarde in één zin",
            "current": "(none)", "proposed": None,
            "html_patch": f"""<p class="page-subhead">{hero.get('subhead')}</p>""",
            "category": "body", "severity": 1, "impact": 3, "effort": 1, "priority": 4.0, "patchable": True
        })
    vps = blocks.get("value_props") or []
    if vps:
        lis = "".join([f"<li><strong>{vp.get('title','')}</strong> — {vp.get('desc','')}</li>" for vp in vps[:4]])
        patches.append({
            "url": url, "field": "value_props",
            "problem": "Add 3–4 concrete value props" if lang == "en" else "Voeg 3–4 concrete voordelen toe",
            "current": "(none)", "proposed": None,
            "html_patch": f"""<section class="value-props"><ul>{lis}</ul></section>""",
            "category": "body", "severity": 2, "impact": 4, "effort": 2, "priority": 5.0, "patchable": True
        })
    steps = blocks.get("steps") or []
    if steps:
        lis = "".join([f"<li>{s}</li>" for s in steps[:5]])
        patches.append({
            "url": url, "field": "steps",
            "problem": "Clarify process in 3–5 steps" if lang == "en" else "Verduidelijk proces in 3–5 stappen",
            "current": "(none)", "proposed": None,
            "html_patch": f"""<section class="steps"><ol>{lis}</ol></section>""",
            "category": "body", "severity": 1, "impact": 3, "effort": 1, "priority": 3.5, "patchable": True
        })
    proof = blocks.get("proof") or []
    if proof:
        lis = "".join([f"<li>{p}</li>" for p in proof[:3]])
        patches.append({
            "url": url, "field": "proof",
            "problem": "Add social proof" if lang == "en" else "Voeg social proof toe",
            "current": "(none)", "proposed": None,
            "html_patch": f"""<section class="proof"><ul>{lis}</ul></section>""",
            "category": "body", "severity": 1, "impact": 3, "effort": 1, "priority": 3.5, "patchable": True
        })
    ctas = blocks.get("ctas") or []
    if ctas:
        btns = "".join([f"""<a href="{c.get('href_hint','#')}" class="btn">{c.get('label','')}</a>""" for c in ctas[:3]])
        patches.append({
            "url": url, "field": "ctas",
            "problem": "Calls to action" if lang == "en" else "CTA’s toevoegen",
            "current": "(none)", "proposed": None,
            "html_patch": f"""<div class="ctas">{btns}</div>""",
            "category": "body", "severity": 1, "impact": 3, "effort": 1, "priority": 3.5, "patchable": True
        })
    return patches

# -----------------------------
# Scoring
# -----------------------------
_EXPECTED_SCHEMA = {
    "home": {"website", "organization"},
    "about": {"aboutpage", "organization", "person"},
    "service": {"service"},
    "pricing": {"product", "offer"},
    "faq": {"faqpage"},
    "contact": {"contactpage"},
    "legal": set(),
    "blog_post": {"article", "blogposting"},
    "listing": {"collectionpage", "blog"},
    "other": set(),
}

def _has_expected_schema(jsonld_types: List[str], page: Dict[str, Any], page_type: str) -> bool:
    types = {t.lower() for t in (jsonld_types or [])}
    expected = _EXPECTED_SCHEMA.get(page_type, set())
    if any(t in types for t in expected):
        return True
    # fallback: via JSON-LD objects
    for obj in _jsonld_objects(page):
        for t in _type_lower(obj.get("@type")):
            if t in expected:
                return True
    return False

def _title_ok(s: str) -> bool:
    s = (s or "").strip()
    return 30 <= len(s) <= 65

def _meta_ok(s: str) -> bool:
    s = (s or "").strip()
    return 140 <= len(s) <= 160

def _canonical_ok(url: str, canon: str) -> bool:
    if not canon:
        return False
    return _norm_url(url) == _norm_url(canon)

def _has_primary_cta(page: Dict[str, Any]) -> bool:
    links = [str(x).lower() for x in (page.get("links") or [])]
    hints = ("/contact", "/pricing", "/subscription", "/scan", "/get-started", "mailto:")
    return any(any(h in l for h in hints) for l in links)

def _score_nonfaq_page(page: Dict[str, Any], page_type: str, intro_words: int, has_expected: bool,
                       canonical_ok: bool, title_ok: bool, meta_ok: bool, has_cta_link: bool
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
    # Weights
    score += metrics["intro_20_80w"] * 10
    score += metrics["has_expected_schema"] * 10
    score += metrics["title_in_range"] * 8
    score += metrics["meta_in_range"] * 7
    score += metrics["canonical_ok"] * 5
    score += metrics["cta_present"] * 5

    if not metrics["intro_20_80w"]:
        issues.append("Intro not 20–80 words")
    if not metrics["has_expected_schema"]:
        issues.append("Missing expected schema for page type")
    if not metrics["title_in_range"]:
        issues.append("Title length suboptimal")
    if not metrics["meta_in_range"]:
        issues.append("Meta description length suboptimal")
    if not metrics["canonical_ok"]:
        issues.append("Canonical differs from page URL" if (page.get("canonical") or "") else "Missing canonical")
    if not metrics["cta_present"]:
        issues.append("No primary CTA link detected")

    return min(score, 90), issues, metrics

def _score_faq_page(qas_count: int, has_faq_schema: bool, answers_leq_80w: int) -> Tuple[int, List[str], Dict[str, int]]:
    issues: List[str] = []
    score = 0

    # Q&A coverage (max 40)
    if qas_count >= 3:
        score += 40
    elif qas_count == 2:
        score += 20; issues.append("Too few Q&A (min 3).")
    elif qas_count == 1:
        score += 10; issues.append("Too few Q&A (min 3).")
    else:
        issues.append("No Q&A present.")

    # Schema (max 30)
    if has_faq_schema:
        score += 30
    else:
        issues.append("No FAQPage JSON-LD.")

    # Answer length (max 30)
    if qas_count > 0:
        ratio = answers_leq_80w / float(qas_count)
        score += int(30 * ratio)
        if answers_leq_80w < qas_count:
            issues.append("Some answers >80 words.")

    metrics = {"qas": qas_count, "has_faq_schema": 1 if has_faq_schema else 0, "answers_leq_80w": answers_leq_80w}
    return min(score, 100), issues, metrics

# -----------------------------
# Main
# -----------------------------
def _unique_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bucket: Dict[str, Dict[str, Any]] = {}

    def _score(pp: Dict[str, Any]) -> Tuple[int, int]:
        return (1 if (pp.get("status") == 200) else 0, int(pp.get("word_count") or 0))

    for p in pages or []:
        u = _norm_url(p.get("final_url") or p.get("url") or "")
        if not u:
            continue
        cur = bucket.get(u)
        if (cur is None) or (_score(p) > _score(cur)):
            bucket[u] = p
    return [{**v, "final_url": u, "url": u} for u, v in bucket.items()]

def generate_aeo(conn, job) -> Dict[str, Any]:
    """
    Returns:
    {
      "pages": [
        {
          "url": str, "lang": "en|nl", "type": "faq|home|service|...",
          "score": int, "issues": [str], "metrics": { ... },
          "qas": [{q,a}], "faq_html": str, "faq_jsonld": { ... },
          "content_patches": [ ... ]  # only for non-FAQ
        }, ...
      ],
      "toggles": { ... }
    }
    """
    site_id = job["site_id"]
    payload = (job.get("payload") or {}) if isinstance(job, dict) else {}
    toggles = {**DEFAULT_TOGGLES, **(payload.get("toggles") or {})}

    site_meta = _fetch_site_meta(conn, site_id)
    crawl = _fetch_latest_job(conn, site_id, "crawl") or {}

    default_lang = _autolang(site_meta, crawl)
    pages_in = _unique_pages((crawl.get("pages") or []))
    out_pages: List[Dict[str, Any]] = []

    for p in pages_in:
        if int(p.get("status") or 0) != 200:
            continue

        url = _norm_url(p.get("final_url") or p.get("url") or "")
        title = (p.get("title") or "").strip()
        h1 = (p.get("h1") or "").strip()
        meta = (p.get("meta_description") or "").strip()
        paragraphs = p.get("paragraphs") or []
        jsonld_types = p.get("jsonld_types") or []
        canon = (p.get("canonical") or "").strip()

        lang = _page_language(p, default_lang)
        page_type = _classify_page_type(url, title, h1)

        is_faq = (_has_faq_schema(p) or page_type == "faq") if toggles["faq_mode"] == "strict" else ("/faq" in url or _has_faq_schema(p) or page_type == "faq")

        # Common page signals
        intro = _first_paragraph(paragraphs)
        intro_words = _word_count(intro)
        has_expected = _has_expected_schema(jsonld_types, p, "faq" if is_faq else page_type)
        title_ok = (30 <= len(title) <= 65)
        meta_ok = (140 <= len(meta) <= 160)
        canon_ok = _canonical_ok(url, canon)
        has_cta = _has_primary_cta(p)

        page_obj: Dict[str, Any] = {"url": url, "lang": lang, "type": "faq" if is_faq else page_type}

        if is_faq:
            # 1) structured fields
            qas = _extract_qas_from_structured(p)

            # 2) headings + paragraphs fallback
            if not qas:
                qas = _extract_qas_from_headings_and_paragraphs(p, lang)

            # 3) LLM fallback
            if not qas:
                qas = _llm_qas_from_page(lang, title, h1, " ".join(paragraphs[:8]), "", n=max(3, min(6, toggles["max_qas_faq"])))

            # trim to ≤80w
            qas_trimmed: List[Dict[str, str]] = []
            answers_leq_80 = 0
            for qa in qas:
                q, a, was_trim = _trim_to_80w(qa["q"], qa["a"])
                if not was_trim:
                    answers_leq_80 += 1
                qas_trimmed.append({"q": q, "a": a})

            score, issues, metrics = _score_faq_page(len(qas_trimmed), _has_faq_schema(p), answers_leq_80)
            page_obj["score"] = score
            page_obj["issues"] = issues
            page_obj["metrics"] = metrics

            top_qas = qas_trimmed[: int(toggles["max_qas_faq"])]
            faq_html = _faq_html_block(top_qas, lang) if top_qas else ""
            faq_jsonld = _faq_jsonld(top_qas) if (top_qas and (not toggles["emit_jsonld_when_visible_only"] or faq_html)) else {}

            page_obj["qas"] = top_qas
            page_obj["faq_html"] = faq_html
            page_obj["faq_jsonld"] = faq_jsonld
            page_obj["content_patches"] = []
        else:
            blocks = _llm_copy_recipe(lang, page_type, title, h1, " ".join(paragraphs[:8]))
            patches = _patch_from_blocks(url, blocks, lang)
            page_obj["content_patches"] = patches

            score, issues, metrics = _score_nonfaq_page(
                p, page_type, intro_words=intro_words, has_expected=has_expected,
                canonical_ok=canon_ok, title_ok=title_ok, meta_ok=meta_ok, has_cta_link=has_cta
            )
            page_obj["score"] = score
            page_obj["issues"] = issues
            page_obj["metrics"] = metrics
            page_obj["qas"] = []
            page_obj["faq_html"] = ""
            page_obj["faq_jsonld"] = {}

        out_pages.append(page_obj)

    return {"pages": out_pages, "toggles": toggles}

__all__ = ["generate_aeo"]
