# general_agent.py
import os, json, time, signal, uuid
from datetime import datetime, timezone
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from psycopg.types.json import Json

from crawl_light import crawl_site
from keywords_agent import generate_keywords
from schema_agent import generate_schema  # AI schema generator

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

import signal as _signal
_signal.signal(_signal.SIGTERM, handle_sigterm)
_signal.signal(_signal.SIGINT, handle_sigterm)

def normalize_output(obj):
    def default(o):
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, uuid.UUID):
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
               SET status='running', started_at=NOW()
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
            safe_output = output if isinstance(output, dict) else {}
            if not safe_output:
                safe_output = {
                    "_aseon": {
                        "note": "forced-nonempty-output",
                        "at": datetime.now(timezone.utc).isoformat()
                    }
                }
            safe_output = normalize_output(safe_output)

            try:
                preview = json.dumps(safe_output)[:400]
            except Exception:
                preview = "<unserializable>"

            log("info", "finish_job_pre_write", job_id=str(job_id), preview=preview)

            cur.execute("""
                UPDATE jobs
                   SET status='done',
                       output=%s,
                       finished_at=NOW(),
                       error=NULL
                 WHERE id=%s
             RETURNING jsonb_typeof(output) AS out_type, output;
            """, (Json(safe_output), job_id))
            wrote = cur.fetchone()
            conn.commit()

            log("info", "finish_job_post_write",
                job_id=str(job_id),
                out_type=(wrote["out_type"] if wrote and "out_type" in wrote else None),
                wrote_preview=(json.dumps(wrote["output"])[:400] if wrote and wrote.get("output") is not None else None)
            )
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
            log("error", "finish_job_failed_write", job_id=str(job_id), error=err_text)

# ---------- Helpers ----------

def get_site_info(conn, site_id):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT s.url, s.language, s.country, a.name AS account_name
              FROM sites s
              JOIN accounts a ON a.id = s.account_id
             WHERE s.id = %s
        """, (site_id,))
        row = cur.fetchone()
        if not row or not row["url"]:
            raise ValueError("Site not found")
        return row["url"], row.get("language"), row.get("country"), row.get("account_name")

def fetch_latest_job_output(conn, site_id, jtype):
    """
    Haalt het output JSON van de meest recente 'done' job voor deze site en type.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT output
              FROM jobs
             WHERE site_id = %s
               AND type = %s
               AND status = 'done'
          ORDER BY COALESCE(finished_at, created_at) DESC
             LIMIT 1
        """, (site_id, jtype))
        row = cur.fetchone()
        return row["output"] if row else None

def build_crawl_context(crawl_output, max_pages=15, max_chars=12000) -> str:
    """
    Zet crawl-resultaat om naar compacte context voor LLM.
    Verwacht structuur zoals: {"pages":[{url,title,meta_description,h1,h2,h3,...}], "summary":..., "quick_wins":[...]}
    Trimt netjes om tokens te sparen.
    """
    if not isinstance(crawl_output, dict):
        return ""
    pages = crawl_output.get("pages") or []
    slim = {
        "pages": [],
        "summary": crawl_output.get("summary"),
        "quick_wins": crawl_output.get("quick_wins"),
    }
    for p in pages[:max_pages]:
        slim["pages"].append({
            "url": p.get("url"),
            "status": p.get("status"),
            "title": p.get("title"),
            "meta_description": p.get("meta_description"),
            "h1": p.get("h1"),
            "h2": p.get("h2"),
            "h3": p.get("h3"),
            "canonical": p.get("canonical"),
            "noindex": p.get("noindex"),
            "nofollow": p.get("nofollow"),
        })
    text = json.dumps(slim, ensure_ascii=False)
    if len(text) > max_chars:
        text = text[:max_chars] + " …(truncated)"
    return text

# ---------- Crawl ----------

def run_crawl(conn, site_id, payload):
    max_pages = int((payload or {}).get("max_pages", 10))
    ua = (payload or {}).get("user_agent") or "AseonBot/0.1 (+https://aseon.ai)"
    start_url, _, _, _ = get_site_info(conn, site_id)
    result = crawl_site(start_url, max_pages=max_pages, ua=ua)
    return result

# ---------- Keywords ----------

def run_keywords(site_id, payload):
    seed = (payload or {}).get("seed", "home")
    market = (payload or {}).get("market", {})
    lang = market.get("language", "en")
    country = market.get("country", "US")
    return generate_keywords(seed, language=lang, country=country, n=int((payload or {}).get("n", 30)))

# ---------- Schema (AI + auto-context uit crawl) ----------

def run_schema(conn, site_id, payload):
    """
    Gebruikt schema_agent.generate_schema(...) met automatische context uit de
    meest recente crawl-output. Payload-context blijft mogelijk en wordt
    gebruikt als crawl ontbreekt.
    """
    site_url, site_lang, site_country, account_name = get_site_info(conn, site_id)

    # payload hints
    biz_type = (payload or {}).get("biz_type", "Organization")
    name = (payload or {}).get("name") or account_name or "Aseon"
    extras = (payload or {}).get("extras") or {}
    explicit_context = (payload or {}).get("context")  # optioneel

    # Auto RAG: pak laatste crawl
    crawl_out = fetch_latest_job_output(conn, site_id, "crawl")
    rag_context = None
    if crawl_out:
        rag_context = "Crawl summary for this site (titles/meta/H1-H3 & issues):\n" + build_crawl_context(crawl_out)
        log("info", "schema_rag_context_attached", size=len(rag_context))
    elif explicit_context:
        rag_context = explicit_context
        log("info", "schema_explicit_context_attached", size=len(rag_context))
    else:
        rag_context = "(no crawl context available; generate minimal valid JSON-LD only with safe, non-fabricated fields)"
        log("info", "schema_no_context", note="fallback_minimal")

    schema_obj = generate_schema(
        biz_type=biz_type,
        site_name=name,
        site_url=site_url,
        language=site_lang,
        extras=extras,
        rag_context=rag_context
    )

    out = {
        "schema": schema_obj,
        "biz_type": biz_type,
        "name": name,
        "url": site_url,
        "language": site_lang,
        "country": site_country,
        "_context_used": "crawl" if crawl_out else ("payload" if explicit_context else "none")
    }
    log("info", "schema_generated_ai", preview=json.dumps(out)[:400])
    return out

# ---------- FAQ (stub for now) ----------

def run_faq(site_id, payload):
    topic = (payload or {}).get("topic") or "general"
    count = int((payload or {}).get("count", 6))
    faqs = []
    for i in range(count):
        faqs.append({
            "q": f"What is {topic} ({i+1})?",
            "a": f"{topic.capitalize()} explained (stub)."
        })
    return {"topic": topic, "faqs": faqs}

# ---------- Report ----------

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
    log("info", "agent_boot",
        poll_interval=POLL_INTERVAL_SEC,
        batch_size=BATCH_SIZE,
        git=os.getenv("RENDER_GIT_COMMIT") or os.getenv("GIT_COMMIT") or "unknown",
        marker="AGENT_VERSION_V10_SCHEMA_RAG_FROM_CRAWL")
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
