import os, json, traceback
from typing import Optional, Literal, Any, Dict, List
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, AnyHttpUrl, Field, EmailStr
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from psycopg.types.json import Json

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")

app = FastAPI(title="Aseon API", version="0.4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
pool = ConnectionPool(conninfo=DATABASE_URL, min_size=1, max_size=5, kwargs={"row_factory": dict_row})

SQL = """
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
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
"""

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

JobType = Literal["crawl","keywords","faq","schema","report"]

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

@app.exception_handler(Exception)
def unhandled(request: Request, exc: Exception):
    print("UNHANDLED ERROR:", repr(exc)); traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail":"Internal Server Error","error":str(exc)})

@app.on_event("startup")
def start():
    with pool.connection() as c:
        c.execute(SQL); c.commit()

@app.get("/healthz")
def health(): 
    try:
        with pool.connection() as c: c.execute("SELECT 1;")
        return {"ok": True, "db": True}
    except Exception:
        return {"ok": False, "db": False}

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

# QoL: output direct endpoint
@app.get("/jobs/{job_id}/output")
def get_job_output(job_id: str):
    with pool.connection() as c, c.cursor() as cur:
        cur.execute("SELECT output FROM jobs WHERE id=%s",(job_id,))
        row = cur.fetchone()
        if not row: raise HTTPException(status_code=404, detail="Job not found")
        return row["output"] or {}

# QoL: latest results per site for a set of types
@app.get("/sites/{site_id}/latest")
def get_site_latest(site_id: str, types: str = Query(..., description="comma-separated list e.g. crawl,keywords,faq,schema")):
    wanted = [t.strip() for t in types.split(",") if t.strip()]
    if not wanted:
        raise HTTPException(status_code=400, detail="No types provided")
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
