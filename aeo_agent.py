# -*- coding: utf-8 -*-
"""
AEO FAQ Agent
-------------
Doel:
- Auditeert ALLEEN een gegeven FAQ-pagina-URL.
- Extraheert Q&A uit JSON-LD en DOM (headings/accordions/definition lists).
- Beoordeelt elke QA (LLM of rule-based fallback).
- Stelt verbeteringen voor (≤ 80 woorden, concreet, non-fluff/promo).
- Bouwt valide FAQPage JSON-LD met verbeterde QA's.
- Levert rijk diagnostisch resultaat voor rapportage.

Env:
- OPENAI_API_KEY (optioneel; zonder key: rule-based fallback)
- LLM_MODEL (default: gpt-4o-mini)
- LLM_TEMPERATURE (default: 0.0)
- HTTP_TIMEOUT_SECONDS (default: 30)
"""

from __future__ import annotations

import os
import re
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Iterable

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, HttpUrl, ValidationError

# ---------------------- Logging ----------------------

LOGGER = logging.getLogger("aeo_agent")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)

# ---------------------- Config ----------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
HTTP_TIMEOUT_SECONDS = int(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))

UA = "aseon-aeo-faq-agent/1.1 (+https://www.aseon.io/)"

MAX_SNIPPET_WORDS = 80
MAX_RAW_ANSWER_WORDS = 120

PROMO_TRIGGERS = [
    r"\b(contact|neem contact|boek|bestel|koop|klik hier|meld je aan|subscribe|sign ?up|demo aanvragen|afrekenen|betaling)\b",
    r"https?://",
]

# ---------------------- Helpers ----------------------

_WS_RE = re.compile(r"\s+")
def norm(x: str) -> str:
    return _WS_RE.sub(" ", (x or "").strip())

def looks_like_question(text: str) -> bool:
    t = norm(text).lower()
    if len(t) < 3:
        return False
    return t.endswith("?") or t.startswith((
        "how ","what ","why ","when ","where ","who ",
        "can ","do ","does ","is ","are ","should ","will ",
        "hoe ","wat ","waarom ","wanneer ","waar ","wie ",
        "kan ","kun ","doet ","is ","zijn ","moet ","zal "
    ))

def truncate_words(text: str, max_words: int) -> str:
    words = norm(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])

def dedupe_by_question(qas: List["QAItem"]) -> List["QAItem"]:
    seen = set()
    out: List[QAItem] = []
    for qa in qas:
        key = norm(qa.question).lower()
        if key and key not in seen:
            seen.add(key)
            out.append(qa)
    return out

def is_promotional(s: str) -> bool:
    s_l = norm(s).lower()
    return any(re.search(p, s_l) for p in PROMO_TRIGGERS)

# ---------------------- Models ----------------------

class QAItem(BaseModel):
    question: str
    answer: str

class QAReview(BaseModel):
    question: str
    answer: str
    is_good: bool
    issues: List[str] = Field(default_factory=list)
    improved_question: Optional[str] = None
    improved_answer: Optional[str] = None
    word_count_answer: int = 0

class FAQAuditResult(BaseModel):
    url: HttpUrl
    found_faq: bool
    qas_extracted: List[QAItem] = Field(default_factory=list)
    reviews: List[QAReview] = Field(default_factory=list)
    suggestions_count: int = 0
    faq_schema_jsonld: Optional[Dict[str, Any]] = None
    notes: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)

# ---------------------- HTTP Fetcher ----------------------

class Fetcher:
    def __init__(self, timeout: int = HTTP_TIMEOUT_SECONDS, retries: int = 2):
        self.timeout = timeout
        self.retries = retries

    def get(self, url: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                with httpx.Client(timeout=self.timeout, follow_redirects=True, headers={"User-Agent": UA}) as client:
                    r = client.get(url)
                    r.raise_for_status()
                    # httpx will auto-detect encoding; rely on r.text
                    return r.text
            except Exception as e:
                last_err = e
                LOGGER.warning("GET failed (attempt %s): %s", attempt + 1, e)
                time.sleep(min(1.5 * (attempt + 1), 4.0))
        raise RuntimeError(f"Failed to fetch URL after retries: {last_err}")

# ---------------------- LLM Client ----------------------

class LLMClient:
    def __init__(self, api_key: str = OPENAI_API_KEY, model: str = LLM_MODEL, temperature: float = LLM_TEMPERATURE):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, system: str, user: str) -> Optional[str]:
        if not self.available():
            return None
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ]
            }
            with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS) as client:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            LOGGER.error("LLM chat error: %s", e)
            return None

# ---------------------- Extractors ----------------------

def extract_qas_from_schema(soup: BeautifulSoup) -> List[QAItem]:
    out: List[QAItem] = []
    for tag in soup.find_all("script", type="application/ld+json"):
        raw = tag.string or ""
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        blocks = data if isinstance(data, list) else [data]
        for b in blocks:
            if not isinstance(b, dict):
                continue
            if str(b.get("@type", "")).lower() != "faqpage":
                continue
            entities = b.get("mainEntity", [])
            entities = entities if isinstance(entities, list) else [entities]
            for e in entities:
                if not isinstance(e, dict):
                    continue
                if str(e.get("@type", "")).lower() != "question":
                    continue
                q = norm(e.get("name") or e.get("text") or "")
                a = ""
                acc = e.get("acceptedAnswer") or {}
                if isinstance(acc, dict):
                    a = norm(acc.get("text") or "")
                if q and a:
                    out.append(QAItem(question=q, answer=a))
    return out

def _nearest_answer_block(node) -> Optional[str]:
    # Vind eerstvolgende betekenisvolle block als antwoord (overslaan van scripts/forms/nav)
    el = node.find_next(lambda el: el and el.name in {"p","div","dd","li","section","article"} and norm(el.get_text(" ", strip=True)))
    if not el:
        return None
    txt = norm(el.get_text(" ", strip=True))
    return txt or None

def extract_qas_from_dom(soup: BeautifulSoup) -> List[QAItem]:
    out: List[QAItem] = []

    # Headings / toggles / summary / buttons met accordion-achtige classes/ARIA
    selectors = [
        "h1","h2","h3","h4","h5","summary","button","dt","strong","b",
        ".faq-question",".accordion-button",".accordion__button","[aria-expanded]"
    ]
    for tag in soup.find_all(selectors):
        q = norm(tag.get_text(" ", strip=True))
        if not looks_like_question(q):
            continue
        ans = _nearest_answer_block(tag)
        if ans:
            out.append(QAItem(question=q, answer=ans))

    # <dl><dt>Q</dt><dd>A</dd></dl>
    for dl in soup.find_all("dl"):
        dts = dl.find_all("dt")
        dds = dl.find_all("dd")
        for dt, dd in zip(dts, dds):
            q = norm(dt.get_text(" ", strip=True))
            a = norm(dd.get_text(" ", strip=True))
            if looks_like_question(q) and a:
                out.append(QAItem(question=q, answer=a))

    return out

def extract_faq(url: str, fetcher: Optional[Fetcher] = None) -> Tuple[List[QAItem], List[str], Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    notes: List[str] = []
    fetcher = fetcher or Fetcher()
    html = fetcher.get(url)
    meta["html_length"] = len(html)
    soup = BeautifulSoup(html, "lxml")

    sch = extract_qas_from_schema(soup)
    dom = extract_qas_from_dom(soup)

    all_qas = dedupe_by_question(sch + dom)

    if not sch:
        notes.append("No FAQPage JSON-LD detected.")
    if not all_qas:
        notes.append("No visible/parseable FAQ detected.")

    meta["counts"] = {"schema": len(sch), "dom": len(dom), "unique": len(all_qas)}
    return all_qas, notes, meta

# ---------------------- Reviewer ----------------------

LLM_SYSTEM = f"""You are an expert AEO auditor for FAQ pages.
Assess each Q&A pair strictly and rewrite if needed.
Rules:
- Keep the original language.
- Answer must be ≤ {MAX_SNIPPET_WORDS} words, factual, non-promotional, no fluff.
- Do not invent claims not supported by the provided answer text.
- If the question isn't a proper question, fix formatting minimally (end with '?').

Return ONLY JSON with keys:
  is_good (bool),
  issues (list of strings),
  improved_question (string or null),
  improved_answer (string or null).
"""

def _rule_review(q: str, a: str) -> Dict[str, Any]:
    issues: List[str] = []
    q_ok = looks_like_question(q)
    if not q_ok:
        issues.append("Vraag is niet duidelijk/geformatteerd als vraag.")
    a_norm = norm(a)
    wc = len(a_norm.split()) if a_norm else 0
    if wc == 0:
        issues.append("Leeg antwoord.")
    if wc > MAX_RAW_ANSWER_WORDS:
        issues.append(f"Antwoord te lang (>{MAX_RAW_ANSWER_WORDS} woorden).")
    if wc < 4:
        issues.append("Antwoord te kort (<4 woorden).")
    if is_promotional(a_norm):
        issues.append("Promotionele/CTA-taal detected.")

    improved_q = None
    if not q_ok:
        q2 = norm(q)
        if not q2.endswith("?"):
            q2 += "?"
        improved_q = q2

    improved_a = None
    if issues:
        a2 = a_norm
        if wc > MAX_SNIPPET_WORDS:
            a2 = truncate_words(a2, MAX_SNIPPET_WORDS)
        improved_a = a2

    return {
        "is_good": not issues,
        "issues": issues,
        "improved_question": improved_q,
        "improved_answer": improved_a,
        "word_count_answer": wc
    }

def _llm_json_parse(s: str) -> Optional[Dict[str, Any]]:
    try:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end+1])
        return json.loads(s)
    except Exception:
        return None

class Reviewer:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def review_one(self, qa: QAItem) -> QAReview:
        # LLM path
        if self.llm.available():
            prompt = f"Question:\n{qa.question}\n\nAnswer:\n{qa.answer}\n\nReturn JSON now with exactly these keys: is_good, issues, improved_question, improved_answer"
            raw = self.llm.chat(LLM_SYSTEM, prompt)
            if raw:
                data = _llm_json_parse(raw)
                if data and set(data.keys()) >= {"is_good","issues","improved_question","improved_answer"}:
                    # enforce ≤80 words hard cap post-LLM
                    improved = data.get("improved_answer") or qa.answer
                    improved = truncate_words(improved, MAX_SNIPPET_WORDS)
                    wc = len(norm(improved).split())
                    return QAReview(
                        question=qa.question,
                        answer=qa.answer,
                        is_good=bool(data.get("is_good")),
                        issues=list(data.get("issues") or []),
                        improved_question=(data.get("improved_question") or None),
                        improved_answer=improved,
                        word_count_answer=wc
                    )
        # fallback
        data = _rule_review(qa.question, qa.answer)
        improved_ans = data["improved_answer"] or qa.answer
        improved_ans = truncate_words(improved_ans, MAX_SNIPPET_WORDS)
        return QAReview(**{
            "question": qa.question,
            "answer": qa.answer,
            "is_good": data["is_good"],
            "issues": data["issues"],
            "improved_question": data["improved_question"],
            "improved_answer": improved_ans,
            "word_count_answer": len(norm(improved_ans).split())
        })

    def review_many(self, qas: List[QAItem]) -> List[QAReview]:
        return [self.review_one(qa) for qa in qas]

# ---------------------- JSON-LD Builder & Validation ----------------------

REQUIRED_FAQ_KEYS = {"@context","@type","mainEntity"}

def build_faqpage_jsonld(items: Iterable[QAReview|QAItem]) -> Dict[str, Any]:
    main = []
    for it in items:
        q = getattr(it, "improved_question", None) or getattr(it, "question", None)
        a = getattr(it, "improved_answer", None) or getattr(it, "answer", None)
        if not q or not a:
            continue
        main.append({
            "@type": "Question",
            "name": q,
            "acceptedAnswer": {"@type": "Answer", "text": a}
        })
    return {"@context":"https://schema.org","@type":"FAQPage","mainEntity":main}

def validate_faq_jsonld(doc: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    if not isinstance(doc, dict):
        return ["FAQ JSON-LD is not an object."]
    missing = REQUIRED_FAQ_KEYS - set(doc.keys())
    if missing:
        issues.append(f"Missing keys: {', '.join(sorted(missing))}")
    main = doc.get("mainEntity", [])
    if not isinstance(main, list) or not main:
        issues.append("mainEntity must be a non-empty list.")
        return issues
    for i, q in enumerate(main, start=1):
        if not isinstance(q, dict):
            issues.append(f"mainEntity[{i}] not an object")
            continue
        if q.get("@type") != "Question":
            issues.append(f"mainEntity[{i}] @type != Question")
        if not q.get("name"):
            issues.append(f"mainEntity[{i}] missing name")
        acc = q.get("acceptedAnswer")
        if not isinstance(acc, dict) or acc.get("@type") != "Answer" or not acc.get("text"):
            issues.append(f"mainEntity[{i}] invalid acceptedAnswer")
        # soft length check
        if len(norm(acc.get("text","")).split()) > MAX_SNIPPET_WORDS:
            issues.append(f"mainEntity[{i}] answer exceeds {MAX_SNIPPET_WORDS} words")
    return issues

# ---------------------- Core API ----------------------

STARTER_FAQ: List[QAItem] = [
    QAItem(question="What is this service and who is it for?", answer="It explains what the product does, who benefits, and which problems it solves."),
    QAItem(question="How does pricing work?", answer="Outline plan structure, billing frequency, and what each plan includes."),
    QAItem(question="How long until I see results?", answer="Set expectations with realistic timelines and the key factors that influence them."),
    QAItem(question="Do you support my CMS or stack?", answer="State supported platforms and any integration notes or limitations."),
    QAItem(question="How can I get support or talk to a human?", answer="Share support hours, response times, and contact options.")
]

def audit_faq_page(url: str) -> FAQAuditResult:
    """Publieke functie: auditeer ALLEEN een FAQ-URL en produceer volledig resultaat."""
    fetcher = Fetcher()
    qas, notes, meta = extract_faq(url, fetcher=fetcher)

    if not qas:
        reviewer = Reviewer()
        reviews = reviewer.review_many(STARTER_FAQ)
        schema = build_faqpage_jsonld(reviews)
        val_issues = validate_faq_jsonld(schema)
        if val_issues:
            notes.append("Generated FAQ JSON-LD validation issues: " + "; ".join(val_issues))
        return FAQAuditResult(
            url=url,
            found_faq=False,
            qas_extracted=[],
            reviews=reviews,
            suggestions_count=sum(1 for r in reviews if not r.is_good or r.improved_answer or r.improved_question),
            faq_schema_jsonld=schema,
            notes=notes + ["No FAQ detected — starter set proposed."],
            meta=meta
        )

    reviewer = Reviewer()
    reviews = reviewer.review_many(qas)
    schema = build_faqpage_jsonld(reviews)
    val_issues = validate_faq_jsonld(schema)
    if val_issues:
        notes.append("Built FAQ JSON-LD validation issues: " + "; ".join(val_issues))

    suggestions = sum(1 for r in reviews if (not r.is_good) or r.improved_answer or r.improved_question)

    return FAQAuditResult(
        url=url,
        found_faq=True,
        qas_extracted=qas,
        reviews=reviews,
        suggestions_count=suggestions,
        faq_schema_jsonld=schema,
        notes=notes,
        meta=meta
    )

# ---------------------- (Optional) CLI ----------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Audit a single FAQ page for AEO.")
    ap.add_argument("--url", required=True)
    args = ap.parse_args()
    res = audit_faq_page(args.url)
    print(json.dumps(res.dict(), ensure_ascii=False, indent=2))
