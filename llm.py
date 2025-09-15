import os
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Request, Body
from pydantic import BaseModel
from openai import OpenAI

from rag_helper import get_rag_context  # jij hebt dit bestand al

router = APIRouter()

OPENAI_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "30"))

def _to_list(v) -> Optional[List[str]]:
    if v is None: return None
    if isinstance(v, list): return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):  return [x.strip() for x in v.split(",") if x.strip()]
    return None

class LLMAnswerRequest(BaseModel):
    query: str
    site_id: Optional[str] = None
    kb_tags: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None
    format: Optional[str] = "markdown"

class LLMAnswerResponse(BaseModel):
    answer: str
    citations: Dict[str, Any]
    used: Dict[str, Any]

def build_prompt(ctx: Dict[str, Any], query: str, out_format: str = "markdown") -> List[Dict[str, str]]:
    site_ctx = ctx.get("site_ctx") or ""
    kb_ctx = ctx.get("kb_ctx") or ""

    system = (
        "You are a precise SEO/AEO assistant. Answer ONLY using the facts from the provided context. "
        "If something is not in the context, say you don’t have enough information. "
        "Cite sources inline as [S1]..[S8] and [K1]..[K6]. "
        "Keep answers crisp and practical."
    )
    style_note = "Write in Markdown." if (out_format or "").lower() == "markdown" else "Write in plain text (no Markdown)."

    user = f"""Query:
{query}

--- SITE CONTEXT ---
{site_ctx}

--- KB CONTEXT ---
{kb_ctx}

Instructions:
- Be succinct.
- Use inline citations like [S2] or [K1].
- Finish with a short Sources list (unique), format: "ID — URL".
- If info is missing, say what’s missing briefly.
- {style_note}
""".strip()

    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]

def _openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)

def _get_db_conn(request: Request):
    pool = getattr(request.app.state, "pool", None)
    if not pool:
        # in main.py gebruiken we ConnectionPool; expose die op app.state
        raise RuntimeError("Database pool not available as app.state.pool")
    return pool.connection()

@router.post("/llm/answer", response_model=LLMAnswerResponse)
def llm_answer(request: Request, body: LLMAnswerRequest = Body(...)):
    try:
        kb_tags = _to_list(body.kb_tags)

        if body.context:
            ctx = body.context
        else:
            if not body.site_id:
                raise HTTPException(status_code=400, detail="site_id required when no context is provided")
            with _get_db_conn(request) as conn:
                ctx = get_rag_context(conn, site_id=body.site_id, query=body.query, kb_tags=kb_tags)

        messages = build_prompt(ctx, body.query, body.format or "markdown")

        client = _openai_client()
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            messages=messages,
            timeout=OPENAI_TIMEOUT_SEC,
        )
        answer = (resp.choices[0].message.content or "").strip()

        return LLMAnswerResponse(
            answer=answer,
            citations={"site": ctx.get("site_citations", []), "kb": ctx.get("kb_citations", [])},
            used={
                "model": OPENAI_MODEL,
                "query": body.query,
                "kb_tags": kb_tags,
                "char_budget": ctx.get("char_budget"),
                "char_used": ctx.get("char_used"),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"llm_answer_failed: {str(e)[:300]}")
