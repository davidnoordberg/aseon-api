# general_agent.py
# V13.2: requeue fix + ingest rollback on failure + low-mem crawl flow compatible
import os, json, time, signal, uuid
from datetime import datetime, timezone
from typing import Any, Optional
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from psycopg.types.json import Json

from crawl_light import crawl_site
from keywords_agent import generate_keywords
from schema_agent import generate_schema
from faq_agent import generate_faqs
from ingest_agent import ingest_crawl_output

POLL_INTERVAL_SEC = float(os.getenv("POLL_INTERVAL_SEC", "2"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
MAX_ERROR_LEN = 500
STALE_MINUTES = int(os.getenv("STALE_MINUTES", "10"))
INGEST_ENABLED = os.getenv("ASEON_INGEST_ENABLED", "true").lower() in ("1","true","yes")
CRAWL_MAX_PAGES_HARD = int(os.getenv("CRAWL_MAX_PAGES_HARD", "12"))

DSN = os.environ["DATABASE_URL"]
running = True

def log(level: str, msg: str, **kwargs):
    print(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level.upper(),
        "msg": msg,
        **kwargs
    }), flush=True)

def handle_sigterm(signum, frame):
    global running
    log("info", "received_shutdown")
    running = False

signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

def normalize_output(obj: Any) -> Any:
    def default(o):
        if isinstance(o, datetime): return o.isoformat()
        if isinstance(o, uuid.UUID): return str(o)
        return str(o)
    return json.loads(json.dumps(obj, default=default))

def requeue_stale(conn):
    """Re-queue jobs that have been 'running' for longer than STALE_MINUTES."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE jobs
               SET status='queued', started_at=NULL, error=NULL
             WHERE status='running'
               AND started_at < NOW() - make_interval(mins => %s)
        """, (STALE_MINUTES,))
        n = cur.rowcount
        conn.commit()
    if n:
        log("warn", "requeued_stale_running_jobs", count=n, older_than_min=STALE_MINUTES)

def claim_one_job(conn):
    with conn.cursor() as cur:
        cur.execute("""
            WITH j AS (
                SELECT id
                  FROM jobs
                 WHERE status='queued'
              ORDER BY created_at
                 LIMIT 1
                 FOR UPDATE SKIP LOCKED
            )
            UPDATE jobs
               SET status='running', started_at=NOW()
              FROM j
             WHERE jobs.id=j.id
         RETURNING jobs.id, jobs.site_id, jobs.type, jobs.payload;
        """)
        row = cur.fetchone()
        conn.commit()
        return row

def finish_job(conn, job_id, ok: bool, output: Optional[dict] = None, err: Optional[Exception] = None):
    with conn.cursor() as cur:
        if ok:
            safe = normalize_output(output if isinstance(output, dict) else {"_aseon":{"note":"empty"}})
            log("info", "finish_job_pre_write", job_id=str(job_id), preview=json.dumps(safe)[:400])
            cur.execute("""
                UPDATE jobs
                   SET status='done', output=%s, finished_at=NOW(), error=NULL
                 WHERE id=%s
            """, (Json(safe), job_id))
            conn.commit()
        else:
            err_text = (str(err) if err else "Unknown error")[:MAX_ERROR_LEN]
            cur.execute("""
                UPDATE jobs
                   SET status='failed', finished_at=NOW(), error=%s
                 WHERE id=%s
            """, (err_text, job_id))
            conn.commit()
            log("error", "finish_job_failed_write", job_id=str(job_id), error=err_text)

# ----- Helpers -----

def get_site_info(conn, site_id):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT s.url, s.language, s.country, a.name AS account_name
              FROM sites s
              JOIN accounts a ON a.id=s.account_id
             WHERE s.id=%s
        """, (site_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError("Site not found")
        return row["url"], row.get("language"), row.get("country"), row.get("account_name")

def fetch_latest_job_output(conn, site_id, jtype: str):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT output
              FROM jobs
             WHERE site_id=%s AND type=%s AND status='done'
          ORDER BY COALESCE(finished_at, created_at) DESC
             LIMIT 1
        """, (site_id, jtype))
        row = cur.fetchone()
        return row["output"] if row else None

def build_crawl_context(crawl_output: dict, max_pages: int = 10, max_chars: int = 9000) -> str:
    pages = (crawl_output or {}).get("pages") or []
    slim = {"pages": [], "summary": (crawl_output or {}).get("summary")}
    for p in pages[:max_pages]:
        slim["pages"].append({
            "url": p.get("final_url") or p.get("url"),
            "title": p.get("title"),
            "meta_description": p.get("meta_description"),
            "h1": p.get("h1"),
            "h2": p.get("h2"),
            "h3": p.get("h3")
        })
    text = json.dumps(slim, ensure_ascii=False)
    return text[:max_chars] + (" …(truncated)" if len(text) > max_chars else "")

# ----- Job runners -----

def run_crawl(conn, site_id, payload):
    max_pages_req = int((payload or {}).get("max_pages", 10))
    max_pages = min(max_pages_req, CRAWL_MAX_PAGES_HARD)
    ua = (payload or {}).get("user_agent") or "AseonBot/0.3 (+https://aseon.ai)"
    start_url, _, _, _ = get_site_info(conn, site_id)

    result = crawl_site(start_url, max_pages=max_pages, ua=ua)

    if INGEST_ENABLED:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass('public.documents') IS NOT NULL AS ok;")
                ok = bool(cur.fetchone()["ok"])
            if ok:
                n = ingest_crawl_output(conn, site_id, result)
                log("info", "ingest_done", chunks=n)
            else:
                log("info", "ingest_skipped", reason="documents_table_missing")
        except Exception as e:
            # BELANGRIJK: rollback zodat de connectie bruikbaar blijft
            try:
                conn.rollback()
            except Exception:
                pass
            log("error", "ingest_failed", error=str(e))
    else:
        log("info", "ingest_disabled_flag")

    return result

def run_keywords(site_id, payload):
    seed = (payload or {}).get("seed", "home")
    market = (payload or {}).get("market", {})
    lang = market.get("language", "en")
    country = market.get("country", "US")
    n = int((payload or {}).get("n", 30))
    return generate_keywords(seed, language=lang, country=country, n=n)

def run_schema(conn, site_id, payload):
    site_url, site_lang, site_country, account_name = get_site_info(conn, site_id)
    biz_type = (payload or {}).get("biz_type", "Organization")
    name = (payload or {}).get("name") or account_name or "Aseon"
    extras = (payload or {}).get("extras") or {}
    explicit_context = (payload or {}).get("context")

    crawl_out = fetch_latest_job_output(conn, site_id, "crawl")
    if crawl_out:
        rag_context = "Crawl summary:\n" + build_crawl_context(crawl_out)
        context_used = "crawl"
    elif explicit_context:
        rag_context = explicit_context
        context_used = "payload"
    else:
        rag_context = "(no crawl context)"
        context_used = "none"

    schema_obj = generate_schema(
        biz_type=biz_type,
        site_name=name,
        site_url=site_url,
        language=site_lang,
        extras=extras,
        rag_context=rag_context
    )

    return {
        "schema": schema_obj,
        "biz_type": biz_type,
        "name": name,
        "url": site_url,
        "language": site_lang,
        "country": site_country,
        "_context_used": context_used
    }

def run_faq(conn, site_id, payload):
    topic = (payload or {}).get("topic") or "site"
    n = int((payload or {}).get("count", 6))
    return generate_faqs(conn, site_id, topic=topic, n=n)

DISPATCH_NEEDS_CONN = {
    "crawl": run_crawl,
    "schema": run_schema,
    "faq": run_faq,
}
DISPATCH_PURE = {
    "keywords": run_keywords,
    "report": lambda site_id, payload: {
        "format": (payload or {}).get("format", "markdown"),
        "report": "# Aseon Report (MVP)\n\n- Crawl: OK\n- Keywords: OK\n- FAQ: OK\n- Schema: OK\n"
    },
}

def process_job(conn, job):
    jtype = job["type"]; site_id = job["site_id"]; payload = job.get("payload") or {}
    log("info", "job_start", id=str(job["id"]), type=jtype, site_id=str(site_id))
    if jtype in DISPATCH_NEEDS_CONN:
        out = DISPATCH_NEEDS_CONN[jtype](conn, site_id, payload)
    elif jtype in DISPATCH_PURE:
        out = DISPATCH_PURE[jtype](site_id, payload)
    else:
        raise ValueError(f"Unknown job type: {jtype}")
    log("info", "job_done", id=str(job["id"]), type=jtype)
    return out

def main():
    global running
    log("info", "agent_boot",
        poll_interval=POLL_INTERVAL_SEC,
        batch_size=BATCH_SIZE,
        git=os.getenv("RENDER_GIT_COMMIT") or os.getenv("GIT_COMMIT") or "unknown",
        marker="AGENT_VERSION_V13_2_INGEST_ROLLBACK_FIX",
        ingest_enabled=INGEST_ENABLED)
    pool = ConnectionPool(DSN, min_size=1, max_size=3, kwargs={"row_factory": dict_row})

    while running:
        try:
            with pool.connection() as conn:
                requeue_stale(conn)
                job = claim_one_job(conn)
                if not job:
                    time.sleep(POLL_INTERVAL_SEC)
                    continue
                try:
                    out = process_job(conn, job)
                    finish_job(conn, job["id"], True, out, None)
                except Exception as e:
                    log("error", "job_failed", id=str(job["id"]), error=str(e))
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    finish_job(conn, job["id"], False, None, e)
        except Exception as loop_err:
            log("error", "loop_error", error=str(loop_err))
            time.sleep(POLL_INTERVAL_SEC)
    log("info", "agent_exit")

if __name__ == "__main__":
    main()
