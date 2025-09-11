import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr, HttpUrl
import psycopg
from psycopg_pool import ConnectionPool

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")

pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=5, kwargs={"autocommit": True})

app = FastAPI(title="Aseon API", version="0.1.0")

# ---------- Pydantic modellen ----------
class AccountIn(BaseModel):
    name: str
    email: EmailStr

class SiteIn(BaseModel):
    account_id: str
    url: HttpUrl
    language: Optional[str] = None
    country: Optional[str] = None

class JobIn(BaseModel):
    site_id: str
    type: str                # bv. 'crawl', 'keywords', 'faq', 'report'
    payload: Optional[dict] = None

# ---------- Health ----------
@app.get("/healthz")
def healthz():
    try:
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute("select 1;")
            _ = cur.fetchone()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------- Accounts ----------
@app.post("/accounts")
def create_account(body: AccountIn):
    q = """
        INSERT INTO accounts (name, email)
        VALUES (%s, %s)
        RETURNING id, name, email, created_at;
    """
    with pool.connection() as conn, conn.cursor() as cur:
        try:
            cur.execute(q, (body.name, body.email))
            row = cur.fetchone()
            return {"id": row[0], "name": row[1], "email": row[2], "created_at": row[3]}
        except psycopg.errors.UniqueViolation:
            raise HTTPException(status_code=409, detail="Email already exists")

# ---------- Sites ----------
@app.post("/sites")
def create_site(body: SiteIn):
    q = """
        INSERT INTO sites (account_id, url, language, country)
        VALUES (%s, %s, %s, %s)
        RETURNING id, account_id, url, language, country, created_at;
    """
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(q, (body.account_id, str(body.url), body.language, body.country))
        row = cur.fetchone()
        return {
            "id": row[0], "account_id": row[1], "url": row[2],
            "language": row[3], "country": row[4], "created_at": row[5]
        }

# ---------- Jobs (queue) ----------
@app.post("/jobs")
def enqueue_job(body: JobIn):
    q = """
        INSERT INTO jobs (site_id, type, payload, status)
        VALUES (%s, %s, %s::jsonb, 'queued')
        RETURNING id, site_id, type, status, created_at;
    """
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(q, (body.site_id, body.type, psycopg.adapters.Json(body.payload or {})))
        row = cur.fetchone()
        return {
            "id": row[0], "site_id": row[1], "type": row[2],
            "status": row[3], "created_at": row[4]
        }
