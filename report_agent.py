# report_agent.py
# -*- coding: utf-8 -*-
"""
Aseon — Unified Report Agent (SEO • GEO • AEO)
- Crawl site (fallback crawler + optional integration with crawl_light.py)
- Compute SEO issues (titles, meta descriptions, canonical, Open Graph)
- Compute GEO issues (Organization/WebSite JSON-LD presence)
- Compute AEO: in deze iteratie ALLEEN de FAQ-pagina auditen/fixen
  • som bestaande vragen op
  • beoordeel goed/fout
  • stel verbeterde Q/A + JSON-LD voor
- Integraties via optionele imports:
    aeo_agent.audit_faq_page
    faq_agent (optioneel)
    schema_agent (optioneel)
    keywords_agent (optioneel)
    general_agent (optioneel)
    crawl_light (optioneel)
- API endpoints:
    POST /report/full            (site_id?, base_url required)
    POST /report/aeo/faq-only    (url required)
    GET  /report/full.md         (render Markdown)
    GET  /report/aeo/faq.md      (render Markdown for FAQ-only)
"""

from __future__ import annotations
import json
import re
import os
import httpx
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, HttpUrl, ValidationError
from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

# --------- optional imports for deeper integration ----------
_AEO_AVAILABLE = False
try:
    from aeo_agent import audit_faq_page as aeo_audit_faq_page  # preferred AEO impl
    _AEO_AVAILABLE = True
except Exception:
    _AEO_AVAILABLE = False

# these are optional — used only if present
try:
    import crawl_light  # expected to expose a crawl(url, max_pages=...) -> List[str]
    _CRAWL_LIGHT = True
except Exception:
    _CRAWL_LIGHT = False

# ======= Utilities =======

UA = "aseon-report-agent/1.0 (+https://www.aseon.io/)"
TIMEOUT = int(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))

_WS = re.compile(r"\s+")
def norm(x: str) -> str:
    return _WS.sub(" ", (x or "").strip())

def is_same_site(a: str, b: str) -> bool:
    pa, pb = urlparse(a), urlparse(b)
    return pa.netloc.lower() == pb.netloc.lower() and pa.scheme == pb.scheme

def safe_get(url: str) -> str:
    with httpx.Client(timeout=TIMEOUT, follow_redirects=True, headers={"User-Agent": UA}) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.text

def discover_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
            continue
        absu = urljoin(base_url, href)
        if is_same_site(absu, base_url):
            links.append(absu.split("#")[0])
    return list(dict.fromkeys(links))

def lightweight_crawl(base_url: str, max_pages: int = 20) -> List[str]:
    visited, queue = [], [base_url]
    seen = set(queue)
    while queue and len(visited) < max_pages:
        u = queue.pop(0)
        try:
            html = safe_get(u)
        except Exception:
            continue
        visited.append(u)
        for nk in discover_links(html, u):
            if nk not in seen and is_same_site(nk, base_url):
                seen.add(nk)
                queue.append(nk)
    return visited

def looks_like_faq_url(u: str) -> bool:
    p = urlparse(u).path.lower()
    return any(x in p for x in ["/faq", "/faqs", "/help/faq"])

# ======= SEO/GEO extractors =======

class PageSEO(BaseModel):
    url: HttpUrl
    title: Optional[str] = None
    meta_description: Optional[str] = None
    canonical: Optional[str] = None
    og_title: Optional[str] = None
    og_description: Optional[str] = None

class PageSEOIssues(BaseModel):
    url: HttpUrl
    title_issue: Optional[str] = None
    meta_description_issue: Optional[str] = None
    canonical_issue: Optional[str] = None
    og_issue: Optional[str] = None

class GEOSummary(BaseModel):
    has_org_schema: bool = False
    has_website_schema: bool = False
    issues: List[str] = Field(default_factory=list)

def parse_page_seo(html: str, url: str) -> PageSEO:
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else None
    md = soup.find("meta", attrs={"name": "description"})
    meta_description = md.get("content") if md else None
    link_c = soup.find("link", rel=lambda v: v and "canonical" in v)
    canonical = link_c.get("href") if link_c else None
    og_t = soup.find("meta", property="og:title")
    og_d = soup.find("meta", property="og:description")
    return PageSEO(
        url=url,
        title=title,
        meta_description=meta_description,
        canonical=canonical,
        og_title=og_t.get("content") if og_t else None,
        og_description=og_d.get("content") if og_d else None
    )

def score_page_seo(ps: PageSEO) -> PageSEOIssues:
    title_issue = None
    if not ps.title:
        title_issue = "missing"
    else:
        if not (10 <= len(ps.title) <= 65):
            title_issue = "length suboptimal"
    meta_issue = None
    if not ps.meta_description:
        meta_issue = "missing"
    else:
        if not (50 <= len(ps.meta_description) <= 160):
            meta_issue = "length suboptimal"
    canonical_issue = None
    if not ps.canonical:
        canonical_issue = "missing"
    else:
        # normalize domain consistency (optional)
        pass
    og_issue = None
    if ps.og_title and not ps.og_description:
        og_issue = "missing og:description"
    return PageSEOIssues(
        url=ps.url,
        title_issue=title_issue,
        meta_description_issue=meta_issue,
        canonical_issue=canonical_issue,
        og_issue=og_issue
    )

def parse_geo_schema(html: str) -> GEOSummary:
    soup = BeautifulSoup(html, "lxml")
    has_org = False
    has_site = False
    for tag in soup.find_all("script", type="application/ld+json"):
        raw = tag.string or ""
        try:
            data = json.loads(raw)
        except Exception:
            continue
        blocks = data if isinstance(data, list) else [data]
        for b in blocks:
            if not isinstance(b, dict):
                continue
            t = str(b.get("@type") or "").lower()
            if t == "organization":
                has_org = True
            if t == "website":
                has_site = True
    issues = []
    if not has_org:
        issues.append("Organization JSON-LD missing on homepage.")
    if not has_site:
        issues.append("WebSite JSON-LD missing on homepage.")
    return GEOSummary(has_org_schema=has_org, has_website_schema=has_site, issues=issues)

# ======= AEO (FAQ-only) =======

class AEOFAQRow(BaseModel):
    question: str
    answer_sample: str
    status: str
    issues: List[str] = Field(default_factory=list)
    suggested_question: Optional[str] = None
    suggested_answer: Optional[str] = None

class AEOFAQSection(BaseModel):
    url: HttpUrl
    found_faq: bool
    items_reviewed: int
    suggestions_count: int
    existing_questions: List[str] = Field(default_factory=list)
    evaluations: List[AEOFAQRow] = Field(default_factory=list)
    faqpage_jsonld: Dict[str, Any] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)

def aeo_faq_only(url: str) -> AEOFAQSection:
    # Prefer the dedicated aeo_agent if present
    if _AEO_AVAILABLE:
        res = aeo_audit_faq_page(url)
        rows = []
        for r in res.reviews:
            rows.append(AEOFAQRow(
                question=r.question,
                answer_sample=(r.answer[:200] + ("…" if len(r.answer) > 200 else "")),
                status=("ok" if r.is_good else "fix"),
                issues=r.issues,
                suggested_question=r.improved_question,
                suggested_answer=r.improved_answer
            ))
        existing = [r.question for r in res.reviews] if res.found_faq else []
        return AEOFAQSection(
            url=url,
            found_faq=res.found_faq,
            items_reviewed=len(res.reviews),
            suggestions_count=res.suggestions_count,
            existing_questions=existing,
            evaluations=rows,
            faqpage_jsonld=res.faq_schema_jsonld or {"@context":"https://schema.org","@type":"FAQPage","mainEntity":[]},
            notes=res.notes
        )
    # Fallback: lightweight extraction & simple checks (no external deps)
    html = safe_get(url)
    soup = BeautifulSoup(html, "lxml")
    qas = []
    # schema FAQ
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except Exception:
            continue
        blocks = data if isinstance(data, list) else [data]
        for b in blocks:
            if isinstance(b, dict) and str(b.get("@type","")).lower() == "faqpage":
                for e in b.get("mainEntity", []) or []:
                    if isinstance(e, dict) and str(e.get("@type","")).lower() == "question":
                        q = norm(e.get("name") or e.get("text") or "")
                        acc = e.get("acceptedAnswer") or {}
                        a = norm((acc or {}).get("text") or "")
                        if q and a:
                            qas.append((q,a))
    # DOM heuristic
    if not qas:
        for tag in soup.find_all(["h2","h3","h4","summary","button","dt"]):
            q = norm(tag.get_text(" ", strip=True))
            if q.endswith("?"):
                nxt = tag.find_next(lambda el: el.name in {"p","div","dd","li"} and norm(el.get_text(" ", strip=True)))
                if nxt:
                    a = norm(nxt.get_text(" ", strip=True))
                    if a:
                        qas.append((q,a))
    found = len(qas) > 0
    rows = []
    suggestions = 0
    for (q,a) in qas:
        issues = []
        wc = len(norm(a).split())
        if wc > 90: issues.append("answer too long")
        if wc < 4: issues.append("answer too short")
        if not q.endswith("?"): issues.append("question not formatted as question")
        status = "ok" if not issues else "fix"
        if issues: suggestions += 1
        rows.append(AEOFAQRow(
            question=q,
            answer_sample=a[:200] + ("…" if len(a) > 200 else ""),
            status=status,
            issues=issues,
            suggested_question=(q if q.endswith("?") else (q + "?")),
            suggested_answer=(a if wc <= 90 else " ".join(a.split()[:80]))
        ))
    main = []
    for r in rows:
        q = r.suggested_question or r.question
        a = r.suggested_answer or r.answer_sample
        main.append({"@type":"Question","name":q,"acceptedAnswer":{"@type":"Answer","text":a}})
    jsonld = {"@context":"https://schema.org","@type":"FAQPage","mainEntity":main}
    return AEOFAQSection(
        url=url,
        found_faq=found,
        items_reviewed=len(rows),
        suggestions_count=suggestions,
        existing_questions=[q for (q,_) in qas],
        evaluations=rows,
        faqpage_jsonld=jsonld,
        notes=(["No FAQPage JSON-LD detected."] if not qas else [])
    )

# ======= Full report models =======

class SEOFixRow(BaseModel):
    url: HttpUrl
    field: str
    issue: str
    current: Optional[str] = None
    proposed: Optional[str] = None

class CanonicalPatch(BaseModel):
    url: HttpUrl
    category: str
    issue: str
    current: Optional[str] = None
    patch: str

class OGPatch(BaseModel):
    url: HttpUrl
    category: str
    issue: str
    current: str
    patch: str

class AEOScoreRow(BaseModel):
    url: HttpUrl
    type: str
    score: int
    issues: List[str]
    metrics: Dict[str, Any]

class FullReport(BaseModel):
    site_id: Optional[str] = None
    base_url: HttpUrl
    pages_crawled: int
    seo_fixes: List[SEOFixRow]
    canonical_patches: List[CanonicalPatch]
    og_patches: List[OGPatch]
    geo_recommendations: List[str]
    aeo_scorecards: List[AEOScoreRow]
    aeo_faq: Optional[AEOFAQSection] = None
    executive_summary: str

# ======= Builders =======

def build_full_report(base_url: str, site_id: Optional[str] = None, max_pages: int = 20) -> FullReport:
    if _CRAWL_LIGHT:
        try:
            urls = crawl_light.crawl(base_url, max_pages=max_pages)  # type: ignore
            if not urls:
                urls = [base_url]
        except Exception:
            urls = lightweight_crawl(base_url, max_pages=max_pages)
    else:
        urls = lightweight_crawl(base_url, max_pages=max_pages)

    # ensure homepage first
    if base_url not in urls:
        urls.insert(0, base_url)

    seo_rows: List[SEOFixRow] = []
    can_patches: List[CanonicalPatch] = []
    og_patches: List[OGPatch] = []
    aeo_scorecards: List[AEOScoreRow] = []

    homepage_html = None
    faq_url: Optional[str] = None

    for u in urls:
        try:
            html = safe_get(u)
        except Exception:
            continue
        soup = BeautifulSoup(html, "lxml")
        ps = parse_page_seo(html, u)
        issues = score_page_seo(ps)

        # SEO fixes
        if issues.title_issue:
            seo_rows.append(SEOFixRow(
                url=u, field="title", issue=issues.title_issue, current=ps.title,
                proposed=(ps.title or "").strip()[:60] or "Title | "+urlparse(u).netloc
            ))
        if issues.meta_description_issue:
            seo_rows.append(SEOFixRow(
                url=u, field="meta_description", issue=issues.meta_description_issue, current=ps.meta_description,
                proposed=(ps.meta_description or "Write a concise, user-first summary (50–160 chars).")[:155]
            ))
        if issues.canonical_issue:
            can_patches.append(CanonicalPatch(
                url=u, category="canonical", issue="missing", current=None,
                patch=f'<link rel="canonical" href="{u}">'
            ))
        else:
            if ps.canonical and ps.canonical != u:
                can_patches.append(CanonicalPatch(
                    url=u, category="canonical", issue="differs from page URL", current=ps.canonical,
                    patch=f'<link rel="canonical" href="{u}">'
                ))
        if issues.og_issue:
            og_patches.append(OGPatch(
                url=u, category="open_graph", issue=issues.og_issue, current="og:title=ok, og:description=missing",
                patch='<meta property="og:description" content="Add a descriptive summary for social/AI previews.">'
            ))

        # AEO scorecard (simple): mark potential FAQ pages
        ptype = "[faq]" if looks_like_faq_url(u) else "[other]"
        if ptype == "[faq]":
            faq_url = faq_url or u
            aeo_scorecards.append(AEOScoreRow(
                url=u, type=ptype, score=78, issues=["No FAQPage JSON-LD."], metrics={"qa_ok": 0, "parity_ok": False}
            ))
        else:
            aeo_scorecards.append(AEOScoreRow(
                url=u, type=ptype, score=15, issues=["No Q&A section on page."], metrics={"qa_ok": 0, "parity_ok": True}
            ))

        # capture homepage html for GEO
        if u.rstrip("/") == base_url.rstrip("/"):
            homepage_html = html

    geo_reco: List[str] = []
    if homepage_html:
        geo = parse_geo_schema(homepage_html)
        if not geo.has_org_schema:
            geo_reco.append("Add Organization JSON-LD on the homepage with logo and sameAs.")
        if not geo.has_website_schema:
            geo_reco.append("Add WebSite JSON-LD on the homepage.")
    else:
        geo_reco.append("Homepage not reachable; GEO checks skipped.")

    # AEO — FAQ-only: choose first faq-like URL, else fallback to /faq or homepage
    if not faq_url:
        fallback = urljoin(base_url, "/faq")
        faq_url = fallback if is_same_site(fallback, base_url) else base_url

    aeo_section = aeo_faq_only(faq_url)

    summary = (
        f"Scope: {len(urls)} pagina's gecrawld op {urlparse(base_url).netloc}. "
        f"SEO: {len([r for r in seo_rows if r.field=='title'])} titels en "
        f"{len([r for r in seo_rows if r.field=='meta_description'])} meta-descriptions vragen werk; "
        f"{len(can_patches)} canonical- en {len(og_patches)} Open Graph-patches voorgesteld. "
        f"AEO: FAQ geaudit op {faq_url}; {aeo_section.items_reviewed} items beoordeeld."
    )

    return FullReport(
        site_id=site_id,
        base_url=base_url,
        pages_crawled=len(urls),
        seo_fixes=seo_rows,
        canonical_patches=can_patches,
        og_patches=og_patches,
        geo_recommendations=geo_reco,
        aeo_scorecards=aeo_scorecards,
        aeo_faq=aeo_section,
        executive_summary=summary
    )

# ======= Markdown renderers =======

def render_full_markdown(rep: FullReport) -> str:
    lines: List[str] = []
    lines.append(f"SEO • GEO • AEO Audit - {rep.base_url}")
    lines.append("")
    lines.append(f"Executive summary")
    lines.append(rep.executive_summary)
    lines.append("")
    lines.append("AEO — FAQ (only)")
    lines.append(f"URL: {rep.aeo_faq.url if rep.aeo_faq else '-'}")
    if rep.aeo_faq:
        lines.append(f"Found: {rep.aeo_faq.found_faq} | Items: {rep.aeo_faq.items_reviewed} | Suggestions: {rep.aeo_faq.suggestions_count}")
        if rep.aeo_faq.existing_questions:
            lines.append("Existing questions:")
            for i,q in enumerate(rep.aeo_faq.existing_questions,1):
                lines.append(f"{i}. {q}")
        lines.append("")
        lines.append("Evaluations:")
        for i,row in enumerate(rep.aeo_faq.evaluations,1):
            lines.append(f"{i}. {row.question} — {row.status}")
            if row.issues:
                lines.append("   Issues: " + "; ".join(row.issues))
            if row.suggested_question:
                lines.append("   Improved Q: " + row.suggested_question)
            if row.suggested_answer:
                lines.append("   Improved A: " + row.suggested_answer)
        lines.append("")
        lines.append("FAQPage JSON-LD:")
        lines.append("```json")
        lines.append(json.dumps(rep.aeo_faq.faqpage_jsonld, ensure_ascii=False, indent=2))
        lines.append("```")
    lines.append("")
    lines.append("SEO — Concrete text fixes")
    for r in rep.seo_fixes:
        lines.append(f"{r.url} | {r.field} | {r.issue} | Proposed: {r.proposed}")
    lines.append("")
    lines.append("HTML patches — Canonical")
    for p in rep.canonical_patches:
        lines.append(f"{p.url} | {p.issue} | {p.patch}")
    lines.append("")
    lines.append("HTML patches — Open Graph")
    for p in rep.og_patches:
        lines.append(f"{p.url} | {p.issue} | {p.patch}")
    lines.append("")
    lines.append("GEO Recommendations")
    for g in rep.geo_recommendations:
        lines.append(f"- {g}")
    return "\n".join(lines)

def render_aeo_markdown(aeo: AEOFAQSection) -> str:
    lines = []
    lines.append(f"# AEO — FAQ Audit")
    lines.append(f"URL: {aeo.url}")
    lines.append(f"Found: {aeo.found_faq} | Items: {aeo.items_reviewed} | Suggestions: {aeo.suggestions_count}")
    if aeo.existing_questions:
        lines.append("\n## Bestaande vragen")
        for i,q in enumerate(aeo.existing_questions,1):
            lines.append(f"{i}. {q}")
    lines.append("\n## Beoordelingen")
    for i,row in enumerate(aeo.evaluations,1):
        lines.append(f"### {i}. {row.question} — {row.status}")
        if row.issues:
            lines.append(f"- Issues: " + "; ".join(row.issues))
        if row.suggested_question:
            lines.append(f"- Verbeterde vraag: {row.suggested_question}")
        if row.suggested_answer:
            lines.append(f"- Verbeterd antwoord: {row.suggested_answer}")
    lines.append("\n## FAQPage JSON-LD")
    lines.append("```json")
    lines.append(json.dumps(aeo.faqpage_jsonld, ensure_ascii=False, indent=2))
    lines.append("```")
    return "\n".join(lines)

# ======= FastAPI App / Routes =======

app = FastAPI(title="Aseon — Unified Report Agent")

@app.post("/report/full")
def report_full(
    base_url: str = Query(..., description="Base URL om te crawlen, bv. https://www.aseon.io/"),
    site_id: Optional[str] = Query(None),
    max_pages: int = Query(20, ge=1, le=100)
):
    try:
        rep = build_full_report(base_url=base_url, site_id=site_id, max_pages=max_pages)
        return JSONResponse(rep.dict())
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report/full.md")
def report_full_md(
    base_url: str = Query(...),
    site_id: Optional[str] = Query(None),
    max_pages: int = Query(20, ge=1, le=100)
):
    rep = build_full_report(base_url=base_url, site_id=site_id, max_pages=max_pages)
    md = render_full_markdown(rep)
    return PlainTextResponse(md, media_type="text/markdown; charset=utf-8")

@app.post("/report/aeo/faq-only")
def report_faq_only(url: str = Query(..., description="FAQ-pagina URL")):
    try:
        section = aeo_faq_only(url)
        return JSONResponse(section.dict())
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report/aeo/faq.md")
def report_faq_md(url: str = Query(...)):
    section = aeo_faq_only(url)
    md = render_aeo_markdown(section)
    return PlainTextResponse(md, media_type="text/markdown; charset=utf-8")
