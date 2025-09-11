# app/main.py
import os
import json
import asyncio
from typing import Optional, Literal, Any, Dict

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, AnyHttpUrl, Field, EmailStr
from fastapi.middleware.cors import CORSMiddleware
import asyncpg

DATABASE_URL = os.getenv("DATABASE_URL")

app = FastAPI(title="Aseon API", version="0.1.0")

# CORS (pas aan naar je eigen domeinen als je wilt)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pool: Optional[asyncpg.pool.Pool] = None


# ---------- SQL helpers ----------
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

# kleine util om url consistent te maken
def normalize_url(u: str) -> str:
    if not u:
        return u
    u = u.strip()
    # voeg https toe als er geen schema is
    if not u.startswith("http://") and not u.startswith("https://"):
        u = "https://" + u
    # trailing slash voor consistentie
    if not u.endswith("/"):
        u = u + "/"
    return u


# ---------- Pydantic modellen ----------
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


# ---------- lifecycle ----------
@app.on_event("startup")
async def on_startup():
    global pool
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)

    # maak tabellen indien nodig
    async with pool.acquire() as conn:
        await conn.execute(CREATE_TABLES_SQL)


@app.on_event("shutdown")
async def on_shutdown():
    global pool
    if pool:
        await pool.close()


# ---------- health ----------
@app.get("/healthz")
async def healthz():
    # check db snel
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1;")
    except Exception:
        return {"ok": False, "db": False}
    return {"ok": True, "db": True}


# ---------- accounts ----------
@app.post("/accounts", response_model=AccountOut)
async def create_account(body: AccountIn):
    async with pool.acquire() as conn:
        # bestaat email al?
        exists = await conn.fetchrow("SELECT id FROM accounts WHERE email=$1", body.email)
        if exists:
            # voor heldere feedback 409
            raise HTTPException(status_code=409, detail="Email already exists")

        row = await conn.fetchrow(
            """
            INSERT INTO accounts (name, email)
            VALUES ($1, $2)
            RETURNING id, name, email, created_at
            """,
            body.name, body.email
        )
        return dict(row)


@app.get("/accounts", response_model=AccountOut)
async def get_account_by_email(email: EmailStr = Query(..., description="Lookup by email")):
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, name, email, created_at FROM accounts WHERE email=$1", email
        )
        if not row:
            raise HTTPException(status_code=404, detail="Account not found")
        return dict(row)


# ---------- sites ----------
@app.post("/sites", response_model=SiteOut)
async def create_site(body: SiteIn):
    async with pool.acquire() as conn:
        # check account bestaat
        acc = await conn.fetchval("SELECT 1 FROM accounts WHERE id=$1", body.account_id)
        if not acc:
            raise HTTPException(status_code=400, detail="account_id does not exist")

        url_norm = normalize_url(body.url)

        row = await conn.fetchrow(
            """
            INSERT INTO sites (account_id, url, language, country)
            VALUES ($1, $2, $3, $4)
            RETURNING id, account_id, url, language, country, created_at
            """,
            body.account_id, url_norm, body.language.lower(), body.country.upper()
        )
        # AnyHttpUrl validatie forceert een http(s) url
        out = dict(row)
        # Pydantic valideert bij response_model
        return out


# ---------- jobs ----------
@app.post("/jobs", response_model=JobOut)
async def create_job(body: JobIn):
    async with pool.acquire() as conn:
        # check site bestaat
        site_exists = await conn.fetchval("SELECT 1 FROM sites WHERE id=$1", body.site_id)
        if not site_exists:
            raise HTTPException(status_code=400, detail="site_id does not exist")

        row = await conn.fetchrow(
            """
            INSERT INTO jobs (site_id, type, payload)
            VALUES ($1, $2, $3)
            RETURNING id, site_id, type, payload, status, created_at, started_at, finished_at, error
            """,
            body.site_id, body.type, json.dumps(body.payload) if body.payload else None
        )
        return dict(row)


# (optioneel) job ophalen
@app.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(job_id: str):
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, site_id, type, payload, status, created_at, started_at, finished_at, error
            FROM jobs WHERE id=$1
            """,
            job_id
        )
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(row)
