# general_agent.py
import os, json, time, signal, uuid
from datetime import datetime, timezone
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from psycopg.types.json import Json

from crawl_light import crawl_site
from keywords_agent import generate_keywords
from schema_agent import generate_schema  # <-- echte AI schema

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
        return row["url"], row.get("account_name"), row.get("language"), row.get("country")

def latest_job_output(conn, site_id, jtype):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT output
              FROM jobs
             WHERE site_id = %s AND type = %s AND status = 'done'
             ORDER BY created_at DESC
             LIMIT 1
        """, (site_id, jtype))
        r = cur.fetchone()
        return (r or {}).get("output") if r else None

def documents_count(conn, site_id):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM documents WHERE site_id=%s", (site_id,))
            r = cur.fetchone()
            return int((r or {}).get("n") or 0)
    except Exception:
        return 0

def build_crawl_context(crawl_out: dict, max_pages: int = 8) -> str | None:
    if not crawl_out: return None
    pages = (crawl_out or {}).get("pages") or []
    items = []
    for p in pages[:max_pages]:
        frag = {
            "url": p.get("final_url") or p.get("url"),
            "title": p.get("title"),
            "h1": p.get("h1"),
            "h2": p.get("h2"),
            "h3": p.get("h3"),
            "meta_description": p.get("meta_description"),
        }
        items.append(frag)
    ctx = {
        "source": "crawl",
        "pages": items
    }
    return json.dumps(ctx, ensure_ascii=False)

def pick_context(conn, site_id, use_flag: str | None) -> tuple[str | None, str]:
    """
    Context preference:
      - "documents": use RAG (documents table) if available
      - "crawl":     use latest crawl summary (titles/h1/h2/h3/descriptions)
      - "auto"/None: prefer documents > crawl > none
      - anything else: none
    """
    use = (use_flag or "auto").lower()
    if use not in ("documents","crawl","auto","none"):
        use = "auto"

    if use == "documents" or use == "auto":
        n = documents_count(conn, site_id)
        if n > 0:
            return json.dumps({"source":"documents","note":f"{n} chunks available"}), "documents"

    if use == "crawl" or use == "auto":
        crawl = latest_job_output(conn, site_id, "crawl")
        ctx = build_crawl_context(crawl)
        if ctx:
            return ctx, "crawl"

    return None, "none"

# ---------- Job runners ----------

def run_crawl(conn, site_id, payload):
    max_pages = int((payload or {}).get("max_pages", 10))
    ua = (payload or {}).get("user_agent") or "AseonBot/0.1 (+https://aseon.ai)"
    start_url, _, _, _ = get_site_info(conn, site_id)
    result = crawl_site(start_url, max_pages=max_pages, ua=ua)
    return result

def run_keywords(site_id, payload):
    seed = (payload or {}).get("seed", "home")
    market = (payload or {}).get("market", {})
    lang = market.get("language", "en")
    country = market.get("country", "US")
    return generate_keywords(seed, language=lang, country=country, n=30)

def run_schema(conn, site_id, payload):
    site_url, account_name, language, country = get_site_info(conn, site_id)
    biz_type = (payload or {}).get("biz_type", "Organization")
    extras = (payload or {}).get("extras") or {}
    count = int((payload or {}).get("count", 3))  # default 3 for FAQPage
    use_context = (payload or {}).get("use_context")  # "documents" | "crawl" | "auto" | "none"

    # Choose context
    rag_context, context_used = pick_context(conn, site_id, use_context)

    # Generate schema (AI)
    schema_obj = generate_schema(
        biz_type=biz_type,
        site_name=account_name,
        site_url=site_url,
        language=language,
        extras=extras,
        rag_context=rag_context,
        faq_count=count,
        max_faq_words=80
    )

    # If FAQPage, ensure <= count and <= max words (extra safety)
    if isinstance(schema_obj, dict) and schema_obj.get("@type") == "FAQPage":
        main = schema_obj.get("mainEntity") or []
        main = main[:count]
        cleaned = []
        for item in main:
            if not isinstance(item, dict): continue
            q = item.get("name"); ans = (item.get("acceptedAnswer") or {}).get("text") or ""
            if not q or not ans: continue
            words = " ".join(ans.split()).split(" ")
            if len(words) > 80:
                ans = " ".join(words[:80])
            cleaned.append({
                "@type": "Question",
                "name": q,
                "acceptedAnswer": {"@type":"Answer","text": ans}
            })
        schema_obj["mainEntity"] = cleaned

    out = {
        "schema": schema_obj,
        "biz_type": biz_type,
        "name": account_name,
        "url": site_url,
        "language": language,
        "country": country,
        "_context_used": context_used
    }
    log("info", "schema_generated_ai", preview=json.dumps(out)[:400])
    return out

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

def main():
    global running
    log("info", "agent_boot",
        poll_interval=POLL_INTERVAL_SEC,
        batch_size=BATCH_SIZE,
        git=os.getenv("RENDER_GIT_COMMIT") or os.getenv("GIT_COMMIT") or "unknown",
        marker="AGENT_VERSION_V14_SCHEMA_CONTEXT")
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
