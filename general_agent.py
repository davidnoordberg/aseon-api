# general_agent.py
import os, json, time, signal, uuid
from datetime import datetime, timezone
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from psycopg.types.json import Json

from crawl_light import crawl_site
from keywords_agent import generate_keywords
from schema_agent import generate_schema
from ingest_agent import ingest_crawl_output
from faq_agent import generate_faqs  # echte FAQ agent
import report_agent  # <<< voor echte report-generatie

POLL_INTERVAL_SEC = float(os.getenv("POLL_INTERVAL_SEC", "2"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
MAX_ERROR_LEN = 500
INGEST_ENABLED = os.getenv("INGEST_ENABLED", "true").lower() == "true"

DSN = os.environ["DATABASE_URL"]
running = True

def log(level, msg, **kwargs):
    payload = {"ts": datetime.now(timezone.utc).isoformat(), "level": level.upper(), "msg": msg, **kwargs}
    print(json.dumps(payload), flush=True)

def handle_sigterm(signum, frame):
    global running
    log("info", "received_shutdown")
    running = False

signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

def normalize_output(obj):
    def default(o):
        if isinstance(o, datetime): return o.isoformat()
        if isinstance(o, uuid.UUID): return str(o)
        return str(o)
    return json.loads(json.dumps(obj, default=default))

def claim_one_job(conn):
    with conn.cursor() as cur:
        cur.execute("""
            WITH j AS (
                SELECT id FROM jobs
                 WHERE status='queued'
                 ORDER BY created_at
                 LIMIT 1
                 FOR UPDATE SKIP LOCKED
            )
            UPDATE jobs SET status='running', started_at=NOW()
             FROM j
            WHERE jobs.id=j.id
        RETURNING jobs.id, jobs.site_id, jobs.type, jobs.payload;
        """)
        row = cur.fetchone()
        conn.commit()
        return row

def finish_job(conn, job_id, ok, output=None, err=None):
    with conn.cursor() as cur:
        if ok:
            safe_output = output if isinstance(output, dict) else {}
            if not safe_output:
                safe_output = {"_aseon": {"note": "forced-nonempty-output", "at": datetime.now(timezone.utc).isoformat()}}
            safe_output = normalize_output(safe_output)
            try:
                preview = json.dumps(safe_output)[:400]
            except Exception:
                preview = "<unserializable>"
            log("info", "finish_job_pre_write", job_id=str(job_id), preview=preview)
            cur.execute("""
                UPDATE jobs
                   SET status='done', output=%s, finished_at=NOW(), error=NULL
                 WHERE id=%s
             RETURNING jsonb_typeof(output) AS out_type, output;
            """, (Json(safe_output), job_id))
            cur.fetchone(); conn.commit()
        else:
            err_text = (str(err) if err else "Unknown error")[:MAX_ERROR_LEN]
            cur.execute("""
                UPDATE jobs SET status='failed', finished_at=NOW(), error=%s
                 WHERE id=%s
            """, (err_text, job_id))
            conn.commit()
            log("error", "finish_job_failed_write", job_id=str(job_id), error=err_text)

# ---------- Helpers ----------

def get_site_info(conn, site_id):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT s.url, s.language, s.country, a.name AS account_name
              FROM sites s
              JOIN accounts a ON a.id = s.account_id
             WHERE s.id = %s
        """, (site_id,))
        row = cur.fetchone()
        if not row or not row["url"]:
            raise ValueError("Site not found")
        return row

def get_latest_job_output(conn, site_id, jtype):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT output
              FROM jobs
             WHERE site_id=%s AND type=%s AND status='done'
             ORDER BY finished_at DESC NULLS LAST, created_at DESC
             LIMIT 1
        """, (site_id, jtype))
        r = cur.fetchone()
        return (r or {}).get("output") if r else None

# ---------- Crawl ----------

def run_crawl(conn, site_id, payload):
    max_pages = int((payload or {}).get("max_pages", 10))
    ua = (payload or {}).get("user_agent") or "AseonBot/0.1 (+https://aseon.ai)"
    site = get_site_info(conn, site_id)
    result = crawl_site(site["url"], max_pages=max_pages, ua=ua)

    if INGEST_ENABLED:
        try:
            inserted = ingest_crawl_output(conn, site_id, result)
            log("info", "ingest_done", chunks=inserted)
        except Exception as e:
            log("error", "ingest_failed", error=str(e))
    return result

# ---------- Keywords (AANGEPAST: voortaan met conn/RAG) ----------

def run_keywords(conn, site_id, payload):
    # haal default taal/land uit site
    site = get_site_info(conn, site_id)
    market = (payload or {}).get("market", {}) or {}
    market.setdefault("language", site.get("language") or "en")
    market.setdefault("country",  site.get("country")  or "NL")

    # laat de agent zelf RAG doen (documents -> vector search; fallback crawl)
    payload = dict(payload or {})
    payload["market"] = market
    return generate_keywords(conn, site_id, payload)

# ---------- Schema ----------

def run_schema(conn, site_id, payload):
    site = get_site_info(conn, site_id)
    biz_type   = (payload or {}).get("biz_type", "Organization")
    extras     = (payload or {}).get("extras") or {}
    use_ctx    = (payload or {}).get("use_context", "auto")
    count      = int((payload or {}).get("count", 3))

    faqs_for_schema = None
    context_used = "none"
    if biz_type == "FAQPage" and use_ctx in ("auto", "faq", "documents", "crawl"):
        latest_faq = get_latest_job_output(conn, site_id, "faq")
        if latest_faq and isinstance(latest_faq.get("faqs"), list) and latest_faq["faqs"]:
            faqs_for_schema = latest_faq["faqs"]
            context_used = "faq"

    data = generate_schema(
        biz_type=biz_type,
        site_name=site.get("account_name"),
        site_url=site.get("url"),
        language=site.get("language"),
        extras={**extras, "count": count, "faqs": faqs_for_schema, "use_context": use_ctx}
    )

    return {
        "schema": data,
        "biz_type": biz_type,
        "name": site.get("account_name"),
        "url": site.get("url"),
        "language": site.get("language"),
        "country": site.get("country"),
        "_context_used": context_used if biz_type == "FAQPage" else None
    }

# ---------- FAQ ----------

def run_faq(conn, site_id, payload):
    site = get_site_info(conn, site_id)
    out = generate_faqs(conn, site_id, payload or {})
    out["site"] = {"url": site.get("url"), "language": site.get("language"), "country": site.get("country")}
    return out

# ---------- Report ----------

def run_report(conn, site_id, payload):
    job_stub = {"site_id": site_id, "payload": payload or {}}
    return report_agent.generate_report(conn, job_stub)

# ---------- Dispatcher ----------

def process_job(conn, job):
    jtype = job["type"]; site_id = job["site_id"]; payload = job.get("payload") or {}
    log("info", "job_start", id=str(job["id"]), type=jtype, site_id=str(site_id))

    if jtype == "crawl":
        output = run_crawl(conn, site_id, payload)
    elif jtype == "schema":
        output = run_schema(conn, site_id, payload)
    elif jtype == "keywords":
        output = run_keywords(conn, site_id, payload)  # <<<< gewijzigd
    elif jtype == "faq":
        output = run_faq(conn, site_id, payload)
    elif jtype == "report":
        output = run_report(conn, site_id, payload)
    else:
        raise ValueError(f"Unknown job type: {jtype}")

    log("info", "job_done", id=str(job["id"]), type=jtype)
    return output

def main():
    global running
    log("info", "agent_boot",
        poll_interval=POLL_INTERVAL_SEC, batch_size=BATCH_SIZE,
        git=os.getenv("RENDER_GIT_COMMIT") or os.getenv("GIT_COMMIT") or "unknown",
        marker="AGENT_VERSION_FAQ_SCHEMA_TIED", ingest_enabled=INGEST_ENABLED)
    pool = ConnectionPool(DSN, min_size=1, max_size=4, kwargs={"row_factory": dict_row})

    while running:
        try:
            with pool.connection() as conn:
                job = claim_one_job(conn)
                if not job:
                    time.sleep(POLL_INTERVAL_SEC); continue
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
