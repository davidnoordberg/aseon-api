# aeo_agent.py
# -*- coding: utf-8 -*-
"""
AEO FAQ Agent
-------------
Doel:
- Auditeert ALLEEN een gegeven FAQ-pagina-URL.
- Extraheert Q&A uit JSON-LD en DOM (headings/accordions/definition lists).
- Beoordeelt elke QA (LLM of rule-based fallback).
- Stelt verbeteringen voor (≤ ~80 woorden, concreet, non-fluff).
- Bouwt valide FAQPage JSON-LD met verbeterde QA's.
- Levert rijk diagnostisch resultaat voor rapportage.

Ontwerp:
- LLMClient (verwisselbaar model via env).
- Fetcher met timeouts/retries + UA.
- Extractors (schema + dom) met dedupe + normalisatie.
- Reviewer met LLM + rule-based fallback.
- JSON-LD builder + validator (basis).
- Result objecten (Pydantic) voor downstream usage.

Env:
- OPENAI_API_KEY (optioneel)
- LLM_MODEL (default: gpt-4o-mini)
- LLM_TEMPERATURE (default: 0.0)
- HTTP_TIMEOUT_SECONDS (default: 30)
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable, Callable

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

UA = "aseon-aeo-faq-agent/1.0 (+https://www.aseon.io/)"

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
        "can ","do ","does ","is ","are ","should ","will "
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
        if key not in seen:
            seen.add(key)
            out.append(qa)
    return out

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
    # Vind eerstvolgende betekenisvolle block als antwoord.
    el = node.find_next(lambda el: el.name in {"p","div","dd","li","section","article"} and norm(el.get_text(" ", strip=True)))
    if not el:
        return None
    txt = norm(el.get_text(" ", strip=True))
    return txt or None

def extract_qas_from_dom(soup: BeautifulSoup) -> List[QAItem]:
    out: List[QAItem] = []

    # 1) Headings / toggles / summary
    for tag in soup.find_all(["h1","h2","h3","h4","h5","summary","button","dt","strong","b"]):
        q = norm(tag.get_text(" ", strip=True))
        if not looks_like_question(q):
            continue
        ans = _nearest_answer_block(tag)
        if ans:
            out.append(QAItem(question=q, answer=ans))

    # 2) <dl><dt>Q</dt><dd>A</dd></dl>
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

LLM_SYSTEM = """You are an expert AEO auditor for FAQ pages.
Assess each Q&A pair strictly:
- Question: clear, specific, intent-rich?
- Answer: ≤ ~80 words, factual, user-first, free of fluff/claims, actionable.
Return ONLY JSON with keys: is_good (bool), issues (list[str]), improved_question (string|null), improved_answer (string|null).
Keep the language of the input.
"""

def _rule_review(q: str, a: str) -> Dict[str, Any]:
    issues: List[str] = []
    q_ok = looks_like_question(q)
    if not q_ok:
        issues.append("Vraag is niet duidelijk/intentie-rijk.")
    a_norm = norm(a)
    if not a_norm:
        issues.append("Leeg antwoord.")
    wc = len(a_norm.split())
    if wc > 90:
        issues.append("Antwoord te lang (>90 woorden).")
    if wc < 4:
        issues.append("Antwoord te kort (<4 woorden).")
    lower = a_norm.lower()
    for bad in ["we are the best", "number one", "market leader"]:
        if bad in lower:
            issues.append("Marketing-claim i.p.v. concreet antwoord.")
            break
    improved_q = None
    improved_a = None
    if not q_ok:
        q2 = norm(q)
        if not q2.endswith("?"):
            q2 += "?"
        improved_q = q2
    if issues:
        a2 = a_norm
        if wc > 90:
            a2 = truncate_words(a2, 80)
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
        if self.llm.available():
            prompt = f"Question:\n{qa.question}\n\nAnswer:\n{qa.answer}\n\nReturn JSON now with exactly these keys: is_good, issues, improved_question, improved_answer"
            raw = self.llm.chat(LLM_SYSTEM, prompt)
            if raw:
                data = _llm_json_parse(raw)
                if data and set(data.keys()) >= {"is_good","issues","improved_question","improved_answer"}:
                    wc = len(norm(data.get("improved_answer") or qa.answer).split())
                    return QAReview(
                        question=qa.question,
                        answer=qa.answer,
                        is_good=bool(data.get("is_good")),
                        issues=list(data.get("issues") or []),
                        improved_question=(data.get("improved_question") or None),
                        improved_answer=(data.get("improved_answer") or None),
                        word_count_answer=wc
                    )
        # fallback
        data = _rule_review(qa.question, qa.answer)
        return QAReview(**{
            "question": qa.question,
            "answer": qa.answer,
            "is_good": data["is_good"],
            "issues": data["issues"],
            "improved_question": data["improved_question"],
            "improved_answer": data["improved_answer"],
            "word_count_answer": data["word_count_answer"]
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

# ---------------------- (Optional) Small CLI ----------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Audit a single FAQ page for AEO.")
    ap.add_argument("--url", required=True)
    args = ap.parse_args()
    res = audit_faq_page(args.url)
    print(json.dumps(res.dict(), ensure_ascii=False, indent=2))
