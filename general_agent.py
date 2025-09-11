# general_agent.py
import os, json, time, signal, sys, uuid
from datetime import datetime, timezone
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

POLL_INTERVAL_SEC = float(os.getenv("POLL_INTERVAL_SEC", "2"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))  # MVP: 1 per claim
MAX_ERROR_LEN = 500

DSN = os.environ["DATABASE_URL"]  # verplicht

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
            BEGIN;
            SELECT id, site_id, type, payload
            FROM jobs
            WHERE status = 'queued'
            ORDER BY created_at
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        """)
        row = cur.fetchone()
        if not row:
            cur.execute("COMMIT")
            return None
        cur.execute("""
            UPDATE jobs
            SET status='running', started_at=NOW()
            WHERE id=%s
        """, (row["id"],))
        cur.execute("COMMIT")
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

# ---------- Jobtype stubs ----------

def run_crawl(site_id, payload):
    depth = int((payload or {}).get("depth", 1))
    max_pages = int((payload or {}).get("max_pages", 10))
    notes = [f"stub crawl: depth={depth}, max_pages={max_pages}"]
    return {
        "pages_checked": min(max_pages, 10),
        "status_codes": {"200": 8, "301": 1, "404": 1},
        "notes": notes
    }

def run_keywords(site_id, payload):
    seed = (payload or {}).get("seed") or "home"
    lang = ((payload or {}).get("market") or {}).get("language", "en")
    country = ((payload or {}).get("market") or {}).get("country", "NL")
    kws = [
        f"{seed} tips",
        f"{seed} pricing",
        f"{seed} best practices",
        f"{seed} faq",
        f"{seed} alternatives",
    ]
    groups = {
        "informational": kws[:3],
        "transactional": kws[3:],
    }
    return {"seed": seed, "language": lang, "country": country, "keywords": kws, "groups": groups}

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

def run_schema(site_id, payload):
    page_url = (payload or {}).get("page_url") or "https://example.com/"
    biz_type = (payload or {}).get("biz_type", "Organization")
    schema = {
        "@context": "https://schema.org",
        "@type": biz_type,
        "url": page_url,
        "name": "Aseon Client",
        "description": "Auto-generated schema (MVP)."
    }
    return {"schema": schema}

def run_report(site_id, payload):
    fmt = (payload or {}).get("format", "markdown")
    report_md = "# Aseon Report (MVP)\n\n- Crawl: OK\n- Keywords: OK\n- FAQ: OK\n- Schema: OK\n"
    if fmt == "html":
        html = "<h1>Aseon Report (MVP)</h1><ul><li>Crawl: OK</li><li>Keywords: OK</li><li>FAQ: OK</li><li>Schema: OK</li></ul>"
        return {"format": "html", "report": html}
    return {"format": "markdown", "report": report_md}

DISPATCH = {
    "crawl": run_crawl,
    "keywords": run_keywords,
    "faq": run_faq,
    "schema": run_schema,
    "report": run_report,
}

def process_job(conn, job):
    jtype = job["type"]
    site_id = job["site_id"]
    payload = job.get("payload") or {}
    if jtype not in DISPATCH:
        raise ValueError(f"Unknown job type: {jtype}")
    log("info", "job_start", id=str(job["id"]), type=jtype, site_id=str(site_id))
    output = DISPATCH[jtype](site_id, payload)
    log("info", "job_done", id=str(job["id"]), type=jtype)
    return output

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
