# app/main.py
import os
import json
import traceback
from typing import Optional, Literal, Any, Dict

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, AnyHttpUrl, Field, EmailStr

from psycopg_pool import ConnectionPool
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Json

# ------- Config -------
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")

app = FastAPI(title="Aseon API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # zet later strakker op je domein
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connection pool (sync, stabiel)
pool = ConnectionPool(conninfo=DATABASE_URL, min_size=1, max_size=5, kwargs={"row_factory": dict_row})

# ------- SQL -------
CREATE_TABLES_SQL = """
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
  error TEXT
);
"""

def normalize_url(u: str) -> str:
    if not u:
        return u
    u = u.strip()
    if not u.startswith("http://") and not u.startswith("https://"):
        u = "https://" + u
    if not u.endswith("/"):
        u = u + "/"
    return u

# ------- Models -------
class AccountIn(BaseModel):
    name: str = Field(min_length=1)
    email: EmailStr

class AccountOut(BaseModel):
    id: str
    name: str
    email: EmailStr
    created_at: str

class SiteIn(BaseModel):
    account_id: str
    url: str
    language: str = Field(min_length=2, max_length=5)
    country: str = Field(min_length=2, max_length=2)

class SiteOut(BaseModel):
    id: str
    account_id: str
    url: AnyHttpUrl
    language: str
    country: str
    created_at: str

JobType = Literal["crawl", "keywords", "faq", "schema", "report"]

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

# ------- Error handler (zorgt dat 500 in JSON komt) -------
@app.exception_handler(Exception)
def unhandled_ex_handler(request: Request, exc: Exception):
    # Log naar stdout (Render logs)
    print("UNHANDLED ERROR:", repr(exc))
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error": str(exc)},
    )

# ------- Startup: tabellen maken -------
@app.on_event("startup")
def on_startup():
    with pool.connection() as conn:
        conn.execute(CREATE_TABLES_SQL)
        conn.commit()

# ------- Health -------
@app.get("/healthz")
def healthz():
    try:
        with pool.connection() as conn:
            conn.execute("SELECT 1;")
        return {"ok": True, "db": True}
    except Exception:
        return {"ok": False, "db": False}

# ------- Accounts -------
@app.post("/accounts", response_model=AccountOut)
def create_account(body: AccountIn):
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM accounts WHERE email=%s", (body.email,))
        exists = cur.fetchone()
        if exists:
            raise HTTPException(status_code=409, detail="Email already exists")
        cur.execute(
            """
            INSERT INTO accounts (name, email)
            VALUES (%s, %s)
            RETURNING id, name, email, created_at
            """,
            (body.name, body.email),
        )
        row = cur.fetchone()
        conn.commit()
        return row

@app.get("/accounts", response_model=AccountOut)
def get_account_by_email(email: EmailStr = Query(..., description="Lookup by email")):
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, name, email, created_at FROM accounts WHERE email=%s", (email,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Account not found")
        return row

# ------- Sites -------
@app.post("/sites", response_model=SiteOut)
def create_site(body: SiteIn):
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM accounts WHERE id=%s", (body.account_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=400, detail="account_id does not exist")

        url_norm = normalize_url(body.url)
        cur.execute(
            """
            INSERT INTO sites (account_id, url, language, country)
            VALUES (%s, %s, %s, %s)
            RETURNING id, account_id, url, language, country, created_at
            """,
            (body.account_id, url_norm, body.language.lower(), body.country.upper()),
        )
        row = cur.fetchone()
        conn.commit()
        return row

# ------- Jobs -------
@app.post("/jobs", response_model=JobOut)
def create_job(body: JobIn):
    with pool.connection() as conn, conn.cursor() as cur:
        # check site
        cur.execute("SELECT 1 FROM sites WHERE id=%s", (body.site_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=400, detail="site_id does not exist")

        # payload veilig naar JSONB
        payload_jsonb = Json(body.payload) if body.payload is not None else None

        cur.execute(
            """
            INSERT INTO jobs (site_id, type, payload)
            VALUES (%s, %s, %s)
            RETURNING id, site_id, type, payload, status, created_at, started_at, finished_at, error
            """,
            (body.site_id, body.type, payload_jsonb),
        )
        row = cur.fetchone()
        conn.commit()
        return row

@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str):
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, site_id, type, payload, status, created_at, started_at, finished_at, error
            FROM jobs WHERE id=%s
            """,
            (job_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        return row
