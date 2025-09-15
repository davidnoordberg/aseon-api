import os, json, hashlib, traceback, re
from typing import Optional, Literal, Any, Dict, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, AnyHttpUrl, Field, EmailStr
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from psycopg.types.json import Json
from openai import OpenAI

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Aseon API", version="0.6.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
pool = ConnectionPool(conninfo=DATABASE_URL, min_size=1, max_size=5, kwargs={"row_factory": dict_row})
app.state.pool = pool

_llm_include_ok = False
try:
    from llm import router as llm_router
    app.include_router(llm_router)
    _llm_include_ok = True
except Exception as _e:
    print(json.dumps({"level":"ERROR","msg":"llm_router_import_failed","error":str(_e)[:300]}), flush=True)

def _route_exists(path: str) -> bool:
    try:
        return any(getattr(r, "path", "") == path for r in app.routes)
    except Exception:
        return False

SQL = """
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS accounts (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL,
  email TEXT UNIQUE NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sites (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
  url TEXT NOT NULL,
  language TEXT NOT NULL,
  country TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS jobs (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  site_id UUID NOT NULL REFERENCES sites(id) ON DELETE CASCADE,
  type TEXT NOT NULL,
  payload JSONB,
  status TEXT NOT NULL DEFAULT 'queued',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ,
  error TEXT,
  output JSONB
);

CREATE TABLE IF NOT EXISTS documents (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  site_id UUID NOT NULL REFERENCES sites(id) ON DELETE CASCADE,
  url TEXT NOT NULL,
  language TEXT,
  content TEXT NOT NULL,
  metadata JSONB,
  content_hash TEXT NOT NULL,
  embedding VECTOR(1536),
  last_seen TIMESTAMPTZ DEFAULT NOW(),
  created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS doc_dedup ON documents(site_id, url, content_hash);
CREATE INDEX IF NOT EXISTS doc_by_site ON documents(site_id);

CREATE TABLE IF NOT EXISTS kb_documents (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  source TEXT,
  url TEXT,
  title TEXT,
  tags TEXT[],
  content TEXT NOT NULL,
  embedding VECTOR(1536),
  created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS kb_tags_idx ON kb_documents USING gin (tags);
"""

SQL_ALTER = """
ALTER TABLE kb_documents ADD COLUMN IF NOT EXISTS content_hash TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS kb_dedup ON kb_documents(url, content_hash);
"""

def _maybe_build_vector_indexes(conn) -> None:
    if str(os.getenv("BUILD_VECTOR_INDEXES", "0")).lower() not in ("1","true","yes","on"):
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS doc_vec_idx
                  ON documents USING hnsw (embedding vector_cosine_ops)
                  WITH (m = 8, ef_construction = 64);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS kb_vec_idx
                  ON kb_documents USING hnsw (embedding vector_cosine_ops)
                  WITH (m = 8, ef_construction = 64);
            """)
        conn.commit()
        print(json.dumps({"level":"INFO","msg":"vector_indexes_ready","type":"hnsw"}), flush=True)
    except Exception as e:
        try: conn.rollback()
        except Exception: pass
        print(json.dumps({"level":"WARN","msg":"vector_index_build_failed","error":str(e)}), flush=True)

def _embed(text: str) -> List[float]:
    text = (text or "").strip()
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding

def _hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

def _trim(s: str, max_chars: int = 1200) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s[:max_chars]

def normalize_url(u: str) -> str:
    if not u: return u
    u = u.strip()
    if not u.startswith(("http://","https://")): u = "https://" + u
    if not u.endswith("/"): u += "/"
    return u

def iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if isinstance(dt, datetime) else None

def account_json(r: dict) -> dict:
    return {"id": str(r["id"]), "name": r["name"], "email": r["email"], "created_at": iso(r["created_at"])}

def site_json(r: dict) -> dict:
    return {
        "id": str(r["id"]), "account_id": str(r["account_id"]),
        "url": r["url"], "language": r["language"], "country": r["country"],
        "created_at": iso(r["created_at"])
    }

def job_json(r: dict) -> dict:
    return {
        "id": str(r["id"]),
        "site_id": str(r["site_id"]),
        "type": r["type"],
        "payload": r["payload"],
        "status": r["status"],
        "created_at": iso(r["created_at"]),
        "started_at": iso(r["started_at"]),
        "finished_at": iso(r["finished_at"]),
        "error": r["error"],
        "output": r.get("output")
    }

class AccountIn(BaseModel):
    name: str = Field(min_length=1)
    email: EmailStr

class AccountOut(BaseModel):
    id: str; name: str; email: EmailStr; created_at: str

class SiteIn(BaseModel):
    account_id: str
    url: str
    language: str = Field(min_length=2, max_length=5)
    country: str = Field(min_length=2, max_length=2)

class SiteOut(BaseModel):
    id: str; account_id: str; url: AnyHttpUrl; language: str; country: str; created_at: str

JobType = Literal["crawl","keywords","faq","schema","report","insight"]

class JobIn(BaseModel):
    site_id: str
    type: JobType
    payload: Optional[Dict[str, Any]] = None

class JobOut(BaseModel):
    id: str
    site_id: str
    type: str
    payload: Optional[Dict[str, Any]] = None
    status: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    output: Optional[Dict[str, Any]] = None

class KBDocIn(BaseModel):
    source: Optional[str] = None
    url: Optional[AnyHttpUrl] = None
    title: str
    tags: List[str] = []
    content: str

class KBBulkIn(BaseModel):
    docs: List[KBDocIn]

@app.exception_handler(Exception)
def unhandled(request: Request, exc: Exception):
    print("UNHANDLED ERROR:", repr(exc)); traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail":"Internal Server Error","error":str(exc)})

@app.on_event("startup")
def start():
    with pool.connection() as c:
        c.execute(SQL); c.commit()
        c.execute(SQL_ALTER); c.commit()
        _maybe_build_vector_indexes(c)
    paths = sorted([getattr(r, "path", "") for r in app.routes])
    print(json.dumps({"level":"INFO","msg":"startup_routes","paths":paths}), flush=True)

@app.get("/healthz")
def health():
    try:
        with pool.connection() as c:
            c.execute("SELECT 1;")
        return {"ok": True, "db": True, "llm_answer": _route_exists("/llm/answer")}
    except Exception:
        return {"ok": False, "db": False, "llm_answer": _route_exists("/llm/answer")}

@app.get("/__routes")
def __routes():
    return sorted([getattr(r, "path", "") for r in app.routes])

@app.post("/accounts", response_model=AccountOut)
def create_account(body: AccountIn):
    with pool.connection() as c, c.cursor() as cur:
        cur.execute("""
            INSERT INTO accounts (name,email)
            VALUES (%s,%s)
            ON CONFLICT (email) DO UPDATE SET name=EXCLUDED.name
            RETURNING id,name,email,created_at
        """, (body.name, body.email))
        row = cur.fetchone(); c.commit(); return account_json(row)

@app.get("/accounts", response_model=AccountOut)
def get_account_by_email(email: EmailStr = Query(...)):
    with pool.connection() as c, c.cursor() as cur:
        cur.execute("SELECT id,name,email,created_at FROM accounts WHERE email=%s",(email,))
        row = cur.fetchone()
        if not row: raise HTTPException(status_code=404, detail="Account not found")
        return account_json(row)

@app.post("/sites", response_model=SiteOut)
def create_site(body: SiteIn):
    with pool.connection() as c, c.cursor() as cur:
        cur.execute("SELECT 1 FROM accounts WHERE id=%s",(body.account_id,))
        if not cur.fetchone(): raise HTTPException(status_code=400, detail="account_id does not exist")
        url_norm = normalize_url(body.url)
        cur.execute("""
            INSERT INTO sites (account_id,url,language,country)
            VALUES (%s,%s,%s,%s)
            RETURNING id,account_id,url,language,country,created_at
        """, (body.account_id, url_norm, body.language.lower(), body.country.upper()))
        row = cur.fetchone(); c.commit(); return site_json(row)

@app.post("/jobs", response_model=JobOut)
def create_job(body: JobIn):
    with pool.connection() as c, c.cursor() as cur:
        cur.execute("SELECT 1 FROM sites WHERE id=%s",(body.site_id,))
        if not cur.fetchone(): raise HTTPException(status_code=400, detail="site_id does not exist")
        pj = Json(body.payload) if body.payload is not None else None
        cur.execute("""
            INSERT INTO jobs (site_id,type,payload)
            VALUES (%s,%s,%s)
            RETURNING id,site_id,type,payload,status,created_at,started_at,finished_at,error,output
        """, (body.site_id, body.type, pj))
        row = cur.fetchone(); c.commit(); return job_json(row)

@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str):
    with pool.connection() as c, c.cursor() as cur:
        cur.execute("""
            SELECT id,site_id,type,payload,status,created_at,started_at,finished_at,error,output
            FROM jobs WHERE id=%s
        """, (job_id,))
        row = cur.fetchone()
        if not row: raise HTTPException(status_code=404, detail="Job not found")
        return job_json(row)

@app.get("/jobs/{job_id}/output")
def get_job_output(job_id: str):
    with pool.connection() as c, c.cursor() as cur:
        cur.execute("SELECT output FROM jobs WHERE id=%s",(job_id,))
        row = cur.fetchone()
        if not row: raise HTTPException(status_code=404, detail="Job not found")
        return row["output"] or {}

@app.get("/sites/{site_id}/latest")
def get_site_latest(site_id: str, types: str = Query(..., description="comma-separated e.g. crawl,keywords,faq,schema")):
    wanted = [t.strip() for t in types.split(",") if t.strip()]
    if not wanted: raise HTTPException(status_code=400, detail="No types provided")
    out: Dict[str, Any] = {}
    with pool.connection() as c, c.cursor() as cur:
        for t in wanted:
            cur.execute("""
                SELECT id, output, finished_at
                  FROM jobs
                 WHERE site_id=%s AND type=%s AND status='done'
              ORDER BY COALESCE(finished_at, created_at) DESC
                 LIMIT 1
            """, (site_id, t))
            row = cur.fetchone()
            out[t] = row if row else None
    return out

@app.post("/kb/docs")
def kb_bulk_insert(body: KBBulkIn):
    if not body.docs:
        raise HTTPException(status_code=400, detail="No docs provided")
    inserted = 0; skipped = 0
    with pool.connection() as c, c.cursor() as cur:
        for d in body.docs:
            content = (d.content or "").strip()
            if not content:
                skipped += 1
                continue
            chash = _hash(content)
            try:
                vec = _embed(content)
            except Exception as e:
                print(json.dumps({"level":"ERROR","msg":"kb_embed_failed","error":str(e)[:200]}), flush=True)
                skipped += 1
                continue
            try:
                cur.execute("""
                    INSERT INTO kb_documents (source,url,title,tags,content,embedding,content_hash)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (url, content_hash) DO NOTHING
                """, (d.source, str(d.url) if d.url else None, d.title, d.tags, content, vec, chash))
                if cur.rowcount > 0: inserted += 1
                else: skipped += 1
            except Exception as e:
                print(json.dumps({"level":"ERROR","msg":"kb_insert_failed","error":str(e)[:200]}), flush=True)
                c.rollback(); skipped += 1
        c.commit()
    return {"inserted": inserted, "skipped": skipped}

@app.get("/kb/search")
def kb_search(q: str = Query(..., min_length=2), k: int = 6, tags: Optional[str] = None):
    tag_list: Optional[List[str]] = [t.strip() for t in tags.split(",")] if tags else None
    try:
        qvec = _embed(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")
    sql = "SELECT id, source, url, title, tags, LEFT(content, 800) AS snippet FROM kb_documents"
    params: List[Any] = []
    if tag_list:
        sql += " WHERE tags && %s"
        params.append(tag_list)
    sql += " ORDER BY embedding <-> %s::vector LIMIT %s"
    params.extend([qvec, k])
    with pool.connection() as c, c.cursor() as cur:
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
    return {"q": q, "k": k, "tags": tag_list, "results": rows}

@app.get("/kb")
def kb_list(limit: int = 50, offset: int = 0):
    with pool.connection() as c, c.cursor() as cur:
        cur.execute("""
            SELECT id, source, url, title, tags, created_at
              FROM kb_documents
             ORDER BY created_at DESC
             LIMIT %s OFFSET %s
        """, (limit, offset))
        return {"items": cur.fetchall(), "limit": limit, "offset": offset}

@app.delete("/kb/{kb_id}")
def kb_delete(kb_id: str):
    with pool.connection() as c, c.cursor() as cur:
        cur.execute("DELETE FROM kb_documents WHERE id=%s", (kb_id,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="KB document not found")
        c.commit()
    return {"deleted": kb_id}

def _search_site_docs(conn, site_id: str, query: Optional[str], k: int = 20) -> List[Dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        if query and query.strip():
            try:
                qvec = _embed(query)
                cur.execute("""
                    SELECT url, LEFT(content, 1000) AS snippet, metadata, last_seen, created_at
                      FROM documents
                     WHERE site_id=%s AND embedding IS NOT NULL
                  ORDER BY embedding <-> %s::vector
                     LIMIT %s
                """, (site_id, qvec, k))
                rows = cur.fetchall()
                if rows:
                    return rows
            except Exception as e:
                try: conn.rollback()
                except Exception: pass
                print(json.dumps({"level":"WARN","msg":"site_vec_failed","error":str(e)[:200]}), flush=True)
            try:
                like_pattern = f"%{query}%"
                cur.execute("""
                    SELECT url, LEFT(content, 1000) AS snippet, metadata, last_seen, created_at
                      FROM documents
                     WHERE site_id=%s AND (content ILIKE %s OR url ILIKE %s)
                     LIMIT %s
                """, (site_id, like_pattern, like_pattern, k))
                return cur.fetchall()
            except Exception as e:
                try: conn.rollback()
                except Exception: pass
                print(json.dumps({"level":"ERROR","msg":"site_like_failed","error":str(e)[:200]}), flush=True)
                return []
        else:
            cur.execute("""
                SELECT url, LEFT(content, 1000) AS snippet, metadata, last_seen, created_at
                  FROM documents
                 WHERE site_id=%s
              ORDER BY COALESCE(last_seen, created_at) DESC
                 LIMIT %s
            """, (site_id, k))
            return cur.fetchall()

def _search_kb(conn, query: str, k: int = 6, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        try:
            qvec = _embed(query)
            if tags:
                cur.execute("""
                    SELECT title, url, source, tags, content
                      FROM kb_documents
                     WHERE tags && %s
                  ORDER BY embedding <-> %s::vector
                     LIMIT %s
                """, (tags, qvec, k))
            else:
                cur.execute("""
                    SELECT title, url, source, tags, content
                      FROM kb_documents
                  ORDER BY embedding <-> %s::vector
                     LIMIT %s
                """, (qvec, k))
            rows = cur.fetchall()
            if rows:
                return rows
        except Exception as e:
            try: conn.rollback()
            except Exception: pass
            print(json.dumps({"level":"WARN","msg":"kb_vec_failed","error":str(e)[:200]}), flush=True)
        like_pattern = f"%{query}%"
        if tags:
            cur.execute("""
                SELECT title, url, source, tags, content
                  FROM kb_documents
                 WHERE tags && %s AND (content ILIKE %s OR title ILIKE %s)
                 LIMIT %s
            """, (tags, like_pattern, like_pattern, k))
        else:
            cur.execute("""
                SELECT title, url, source, tags, content
                  FROM kb_documents
                 WHERE (content ILIKE %s OR title ILIKE %s)
                 LIMIT %s
            """, (like_pattern, like_pattern, k))
        return cur.fetchall()

def _build_context(site_rows: List[Dict[str, Any]], kb_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    site_bits, kb_bits = [], []
    site_cites, kb_cites = [], []
    for i, r in enumerate(site_rows[:8]):
        sid = f"S{i+1}"
        site_cites.append({"id": sid, "url": r.get("url")})
        site_bits.append(f"[{sid}] {r.get('url')}\n{_trim(r.get('snippet') or '')}")
    for i, r in enumerate(kb_rows[:6]):
        kid = f"K{i+1}"
        title = r.get("title") or (r.get("source") or "KB")
        kb_cites.append({"id": kid, "url": r.get("url"), "title": title})
        kb_bits.append(f"[{kid}] {title} — {r.get('url')}\n{_trim(r.get('content') or '')}")
    return {
        "site_ctx": "\n\n".join(site_bits),
        "kb_ctx": "\n\n".join(kb_bits),
        "site_citations": site_cites,
        "kb_citations": kb_cites
    }

def _get_rag_context(conn, site_id: str, query: str,
                     k_site: int = 8, k_kb: int = 6,
                     kb_tags: Optional[List[str]] = None) -> Dict[str, Any]:
    srows = _search_site_docs(conn, site_id, query, k=k_site)
    krows = _search_kb(conn, query, k=k_kb, tags=kb_tags)
    return _build_context(srows, krows)

@app.get("/sites/{site_id}/docs")
def list_site_docs(site_id: str,
                   q: Optional[str] = Query(None, description="optional search query"),
                   limit: int = 20):
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    with pool.connection() as c:
        rows = _search_site_docs(c, site_id, q, k=limit)
    out = []
    for r in rows:
        out.append({
            "url": r.get("url"),
            "snippet": r.get("snippet"),
            "last_seen": iso(r.get("last_seen")),
            "created_at": iso(r.get("created_at")),
            "issues": (r.get("metadata") or {}).get("issues")
        })
    return {"items": out, "count": len(out), "site_id": site_id, "q": q}

@app.get("/rag/context")
def rag_context(site_id: str = Query(...),
                q: str = Query(..., min_length=2),
                k_site: int = 8,
                k_kb: int = 6,
                kb_tags: Optional[str] = Query(None, description="comma-separated like 'Schema,SEO'")):
    if k_site < 1 or k_site > 20 or k_kb < 0 or k_kb > 20:
        raise HTTPException(status_code=400, detail="k_site 1..20, k_kb 0..20")
    tags_list = [t.strip() for t in kb_tags.split(",")] if kb_tags else None
    with pool.connection() as c:
        ctx = _get_rag_context(c, site_id, q, k_site=k_site, k_kb=k_kb, kb_tags=tags_list)
    return {"site_id": site_id, "query": q, "kb_tags": tags_list, **ctx}

@app.get("/rag/sample_prompt")
def rag_sample_prompt(
    role: str = "You are a helpful SEO assistant. Use the provided context and cite sources like [S1] or [K2].",
    style: str = "Concise, factual, 1-2 paragraphs."
):
    return {"system": role, "style": style, "how_to_cite": "Cite inline like [S1] or [K2]."}

class _LLMAnswerRequest(BaseModel):
    query: str
    site_id: Optional[str] = None
    kb_tags: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None
    format: Optional[str] = "markdown"

class _LLMAnswerResponse(BaseModel):
    answer: str
    citations: Dict[str, Any]
    used: Dict[str, Any]

def _build_llm_messages(ctx: Dict[str, Any], query: str, out_format: str = "markdown"):
    site_ctx = ctx.get("site_ctx") or ""
    kb_ctx   = ctx.get("kb_ctx") or ""
    system = (
        "You are a precise SEO/AEO assistant. Answer ONLY from the provided context. "
        "Cite inline as [S1]..[S8] and [K1]..[K6]. If info is missing, say so."
    )
    style = "Write in Markdown." if (out_format or "").lower()=="markdown" else "Write in plain text."
    user = f"""Query:
{query}

--- SITE CONTEXT ---
{site_ctx}

--- KB CONTEXT ---
{kb_ctx}

Instructions:
- Be concise and practical.
- Use inline citations like [S2], [K1].
- End with a short 'Sources' list (unique IDs with URLs).
- {style}
"""
    return [{"role":"system","content":system},{"role":"user","content":user}]

if not _route_exists("/llm/answer"):
    @app.post("/llm/answer", response_model=_LLMAnswerResponse)
    def llm_answer_fallback(request: Request, body: _LLMAnswerRequest = Body(...)):
        try:
            kb_tags = [x.strip() for x in (body.kb_tags or []) if x and x.strip()]
            if body.context:
                ctx = body.context
            else:
                if not body.site_id:
                    raise HTTPException(status_code=400, detail="site_id required when no context is provided")
                with request.app.state.pool.connection() as conn:
                    ctx = _get_rag_context(conn, site_id=body.site_id, query=body.query, kb_tags=kb_tags)

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            resp = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                temperature=0.2,
                messages=_build_llm_messages(ctx, body.query, body.format or "markdown"),
                timeout=float(os.getenv("OPENAI_TIMEOUT_SEC", "30")),
            )
            answer = resp.choices[0].message.content.strip()
            return _LLMAnswerResponse(
                answer=answer,
                citations={"site": ctx.get("site_citations", []), "kb": ctx.get("kb_citations", [])},
                used={"model": os.getenv("LLM_MODEL","gpt-4o-mini"), "query": body.query, "kb_tags": kb_tags,
                      "char_budget": ctx.get("char_budget"), "char_used": ctx.get("char_used")}
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"llm_answer_failed(fallback): {str(e)[:300]}")
