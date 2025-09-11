import os, json, time, signal, sys, uuid
from datetime import datetime, timezone
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from crawl_light import crawl_site
from keywords_agent import generate_keywords
# schema_agent importeren we nu NIET, we doen fallback test
# from schema_agent import generate_schema

POLL_INTERVAL_SEC = float(os.getenv("POLL_INTERVAL_SEC", "2"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
MAX_ERROR_LEN = 500

DSN = os.environ["DATABASE_URL"]

running = True

def log(level, msg, **kwargs):
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level.upper(),
        "msg": msg,
        **kwargs,
    }
    print(json.dumps(payload), flush=True)

def handle_sigterm(signum, frame):
    global running
    log("info", "received_shutdown")
    running = False

signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

def normalize_output(obj):
    def default(o):
        if isinstance(o, (datetime,)):
            return o.isoformat()
        if isinstance(o, (uuid.UUID,)):
            return str(o)
        return str(o)
    return json.loads(json.dumps(obj, default=default))

def claim_one_job(conn):
    with conn.cursor() as cur:
        cur.execute("""
            WITH j AS (
                SELECT id
                FROM jobs
                WHERE status = 'queued'
                ORDER BY created_at
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            UPDATE jobs
            SET status = 'running', started_at = NOW()
            FROM j
            WHERE jobs.id = j.id
            RETURNING jobs.id, jobs.site_id, jobs.type, jobs.payload;
        """)
        row = cur.fetchone()
        conn.commit()
        return row

def finish_job(conn, job_id, ok, output=None, err=None):
    with conn.cursor() as cur:
        if ok:
            cur.execute("""
                UPDATE jobs
                SET status='done',
                    output=%s,
                    finished_at=NOW(),
                    error=NULL
                WHERE id=%s
            """, (json.dumps(normalize_output(output or {})), job_id))
        else:
            err_text = (str(err) if err else "Unknown error")[:MAX_ERROR_LEN]
            cur.execute("""
                UPDATE jobs
                SET status='failed',
                    finished_at=NOW(),
                    error=%s
                WHERE id=%s
            """, (err_text, job_id))
        conn.commit()

# ---------- Crawl integration ----------

def get_site_info(conn, site_id):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT s.url, a.name AS account_name
            FROM sites s
            JOIN accounts a ON a.id = s.account_id
            WHERE s.id = %s
        """, (site_id,))
        row = cur.fetchone()
        if not row or not row["url"]:
            raise ValueError("Site not found")
        return row["url"], row.get("account_name")

def run_crawl(conn, site_id, payload):
    max_pages = int((payload or {}).get("max_pages", 10))
    ua = (payload or {}).get("user_agent") or "AseonBot/0.1 (+https://aseon.ai)"
    start_url, _ = get_site_info(conn, site_id)
    result = crawl_site(start_url, max_pages=max_pages, ua=ua)
    return result

# ---------- AI-powered keywords ----------

def run_keywords(site_id, payload):
    seed = (payload or {}).get("seed", "home")
    market = (payload or {}).get("market", {})
    lang = market.get("language", "en")
    country = market.get("country", "US")
    return generate_keywords(seed, language=lang, country=country, n=30)

# ---------- TEMP schema fallback ----------

def run_schema(conn, site_id, payload):
    site_url, account_name = get_site_info(conn, site_id)
    biz_type = (payload or {}).get("biz_type", "Organization")
    name = (payload or {}).get("name") or account_name or "TestSite"

    dummy_schema = {
        "@context": "https://schema.org",
        "@type": biz_type,
        "name": name,
        "url": site_url,
        "note": "This is a dummy schema to verify DB output storage."
    }

    log("info", "schema_generated_dummy", schema=dummy_schema)
    return {"schema": dummy_schema, "biz_type": biz_type, "name": name, "url": site_url}

# ---------- Other jobtypes (stubs) ----------

def run_faq(site_id, payload):
    topic = (payload or {}).get("topic") or "general"
    count = int((payload or {}).get("count", 6))
    faqs = []
    for i in range(count):
        faqs.append({
            "q": f"What is {topic} ({i+1})?",
            "a": f"{topic.capitalize()} explained in a simple, concise way (stub)."
        })
    return {"topic": topic, "faqs": faqs}

def run_report(site_id, payload):
    fmt = (payload or {}).get("format", "markdown")
    report_md = "# Aseon Report (MVP)\n\n- Crawl: OK\n- Keywords: OK\n- FAQ: OK\n- Schema: OK\n"
    if fmt == "html":
        html = "<h1>Aseon Report (MVP)</h1><ul><li>Crawl: OK</li><li>Keywords: OK</li><li>FAQ: OK</li><li>Schema: OK</li></ul>"
        return {"format": "html", "report": html}
    return {"format": "markdown", "report": report_md}

DISPATCH = {
    "keywords": run_keywords,
    "faq": run_faq,
    "report": run_report,
}

# ---------- Job processing ----------

def process_job(conn, job):
    jtype = job["type"]
    site_id = job["site_id"]
    payload = job.get("payload") or {}

    log("info", "job_start", id=str(job["id"]), type=jtype, site_id=str(site_id))

    if jtype == "crawl":
        output = run_crawl(conn, site_id, payload)
    elif jtype == "schema":
        output = run_schema(conn, site_id, payload)
    else:
        if jtype not in DISPATCH:
            raise ValueError(f"Unknown job type: {jtype}")
        output = DISPATCH[jtype](site_id, payload)

    log("info", "job_done", id=str(job["id"]), type=jtype)
    return output

# ---------- Main loop ----------

def main():
    global running
    log("info", "agent_boot", poll_interval=POLL_INTERVAL_SEC, batch_size=BATCH_SIZE)
    pool = ConnectionPool(DSN, min_size=1, max_size=4, kwargs={"row_factory": dict_row})

    while running:
        try:
            with pool.connection() as conn:
                job = claim_one_job(conn)
                if not job:
                    time.sleep(POLL_INTERVAL_SEC)
                    continue
                try:
                    output = process_job(conn, job)
                    finish_job(conn, job["id"], True, output, None)
                except Exception as e:
                    log("error", "job_failed", id=str(job["id"]), error=str(e))
                    finish_job(conn, job["id"], False, None, e)
        except Exception as loop_err:
            log("error", "loop_error", error=str(loop_err))
            time.sleep(POLL_INTERVAL_SEC)

    log("info", "agent_exit")

if __name__ == "__main__":
    main()
