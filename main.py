import os
import json
from typing import Optional, Literal, Any, Dict

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, AnyHttpUrl, Field, EmailStr
from fastapi.middleware.cors import CORSMiddleware

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json  # adapter voor JSON -> jsonb

DATABASE_URL = os.getenv("DATABASE_URL")

app = FastAPI(title="Aseon API", version="0.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Eén async-verbinding die we hergebruiken
db: Optional[psycopg.AsyncConnection] = None


# ---------- SQL ----------
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


# ---------- Pydantic ----------
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
    global db
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is not set")

    # dict_row geeft ons dicts terug i.p.v. tuples
    db = await psycopg.AsyncConnection.connect(DATABASE_URL, row_factory=dict_row)
    async with db.cursor() as cur:
        await cur.execute(CREATE_TABLES_SQL)

@app.on_event("shutdown")
async def on_shutdown():
    global db
    if db:
        await db.close()


# ---------- health ----------
@app.get("/healthz")
async def healthz():
    try:
        async with db.cursor() as cur:
            await cur.execute("SELECT 1;")
            await cur.fetchone()
    except Exception:
        return {"ok": False, "db": False}
    return {"ok": True, "db": True}


# ---------- accounts ----------
@app.post("/accounts", response_model=AccountOut)
async def create_account(body: AccountIn):
    async with db.cursor() as cur:
        await cur.execute("SELECT id FROM accounts WHERE email=%s", (body.email,))
        exists = await cur.fetchone()
        if exists:
            # 409: bestaat al
            raise HTTPException(status_code=409, detail="Email already exists")

        await cur.execute(
            """
            INSERT INTO accounts (name, email)
            VALUES (%s, %s)
            RETURNING id, name, email, created_at
            """,
            (body.name, body.email),
        )
        row = await cur.fetchone()
        # row is dict door dict_row
        row["id"] = str(row["id"])
        row["created_at"] = row["created_at"].isoformat()
        return row

@app.get("/accounts", response_model=AccountOut)
async def get_account_by_email(email: EmailStr = Query(..., description="Lookup by email")):
    async with db.cursor() as cur:
        await cur.execute(
            "SELECT id, name, email, created_at FROM accounts WHERE email=%s", (email,)
        )
        row = await cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Account not found")
        row["id"] = str(row["id"])
        row["created_at"] = row["created_at"].isoformat()
        return row


# ---------- sites ----------
@app.post("/sites", response_model=SiteOut)
async def create_site(body: SiteIn):
    async with db.cursor() as cur:
        await cur.execute("SELECT 1 FROM accounts WHERE id=%s", (body.account_id,))
        acc = await cur.fetchone()
        if not acc:
            raise HTTPException(status_code=400, detail="account_id does not exist")

        url_norm = normalize_url(body.url)

        await cur.execute(
            """
            INSERT INTO sites (account_id, url, language, country)
            VALUES (%s, %s, %s, %s)
            RETURNING id, account_id, url, language, country, created_at
            """,
            (body.account_id, url_norm, body.language.lower(), body.country.upper()),
        )
        row = await cur.fetchone()
        row["id"] = str(row["id"])
        row["account_id"] = str(row["account_id"])
        row["created_at"] = row["created_at"].isoformat()
        return row


# ---------- jobs ----------
@app.post("/jobs", response_model=JobOut)
async def create_job(body: JobIn):
    async with db.cursor() as cur:
        await cur.execute("SELECT 1 FROM sites WHERE id=%s", (body.site_id,))
        site_exists = await cur.fetchone()
        if not site_exists:
            raise HTTPException(status_code=400, detail="site_id does not exist")

        # Gebruik Json() adapter: Python dict -> JSON -> jsonb
        payload_param = Json(body.payload) if body.payload is not None else None

        await cur.execute(
            """
            INSERT INTO jobs (site_id, type, payload)
            VALUES (%s, %s, %s)
            RETURNING id, site_id, type, payload, status, created_at, started_at, finished_at, error
            """,
            (body.site_id, body.type, payload_param),
        )
        row = await cur.fetchone()

        # Defensief decoderen van payload (sommige drivers variëren)
        payload_value = row.get("payload")
        if isinstance(payload_value, (bytes, bytearray, memoryview)):
            try:
                payload_value = json.loads(bytes(payload_value).decode("utf-8"))
            except Exception:
                payload_value = None
        elif isinstance(payload_value, str):
            try:
                payload_value = json.loads(payload_value)
            except Exception:
                pass  # als het al str-json is die je wilt laten staan

        out = {
            "id": str(row["id"]),
            "site_id": str(row["site_id"]),
            "type": row["type"],
            "payload": payload_value,
            "status": row["status"],
            "created_at": row["created_at"].isoformat(),
            "started_at": row["started_at"].isoformat() if row["started_at"] else None,
            "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
            "error": row["error"],
        }
        return out


@app.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(job_id: str):
    async with db.cursor() as cur:
        await cur.execute(
            """
            SELECT id, site_id, type, payload, status, created_at, started_at, finished_at, error
            FROM jobs WHERE id=%s
            """,
            (job_id,),
        )
        row = await cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")

        payload_value = row.get("payload")
        if isinstance(payload_value, (bytes, bytearray, memoryview)):
            try:
                payload_value = json.loads(bytes(payload_value).decode("utf-8"))
            except Exception:
                payload_value = None
        elif isinstance(payload_value, str):
            try:
                payload_value = json.loads(payload_value)
            except Exception:
                pass

        return {
            "id": str(row["id"]),
            "site_id": str(row["site_id"]),
            "type": row["type"],
            "payload": payload_value,
            "status": row["status"],
            "created_at": row["created_at"].isoformat(),
            "started_at": row["started_at"].isoformat() if row["started_at"] else None,
            "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
            "error": row["error"],
        }
