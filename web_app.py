"""
ARKAINBRAIN — AI-Powered Gaming Intelligence Platform
by ArkainGames.com
"""
import html, json, os, secrets, sqlite3, subprocess, time, uuid
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path

os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"  # Disable tracing prompt
os.environ["DO_NOT_TRACK"] = "1"
os.environ["CREWAI_STORAGE_DIR"] = "/tmp/crewai_storage"

# ── Pre-create CrewAI config to prevent interactive tracing prompt ──
for _d in [Path.home() / ".crewai", Path("/tmp/crewai_storage")]:
    _d.mkdir(parents=True, exist_ok=True)
    _cfg = _d / "config.json"
    if not _cfg.exists():
        _cfg.write_text(json.dumps({"tracing_enabled": False, "tracing_disabled": True}))

from flask import Flask, redirect, url_for, session, request, jsonify, send_from_directory, Response
from werkzeug.middleware.proxy_fix import ProxyFix
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)  # Trust Railway's reverse proxy

# XSS protection — escape user-supplied content before rendering in HTML
_esc = html.escape

# sqlite3.Row does not support .get() — use this helper everywhere
def _rget(row, key, default=None):
    """Safe .get() for sqlite3.Row objects."""
    try:
        val = row[key]
        return val if val is not None else default
    except (IndexError, KeyError):
        return default

# ── Stable SECRET_KEY — survives process restarts, gunicorn recycling, deploys ──
# Priority: env var → persisted file → generate-and-save
# Without this, every gunicorn --max-requests restart invalidates ALL sessions.
def _get_or_create_secret_key():
    # 1. Explicit env var — always wins
    env_key = os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY")
    if env_key:
        return env_key
    # 2. Persisted to file — survives process restarts within same container
    key_file = Path(os.getenv("DB_PATH", "arkainbrain.db")).parent / ".flask_secret_key"
    try:
        if key_file.exists():
            stored = key_file.read_text().strip()
            if len(stored) >= 32:
                return stored
    except Exception:
        pass
    # 3. Generate once and save
    new_key = secrets.token_hex(32)
    try:
        key_file.write_text(new_key)
    except Exception:
        pass  # In-memory only if filesystem is truly read-only
    return new_key

app.secret_key = _get_or_create_secret_key()
if not (os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY")):
    print("[WARN] FLASK_SECRET_KEY not set — sessions may not survive Railway redeploys. "
          "Set it in Railway env vars for permanent session persistence.")

# ── Session configuration — persist across browser restarts + devices ──
app.config["PREFERRED_URL_SCHEME"] = "https"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
# Only set Secure=True in production (HTTPS)
if os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RENDER") or os.getenv("FLY_APP_NAME"):
    app.config["SESSION_COOKIE_SECURE"] = True

LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

oauth = OAuth(app)
google = oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID", ""),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET", ""),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = os.getenv("DB_PATH", "arkainbrain.db")

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")     # Concurrent reads + writes
    conn.execute("PRAGMA busy_timeout=5000")     # Wait up to 5s for lock
    return conn

def init_db():
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT, picture TEXT, created_at TEXT DEFAULT (datetime('now')));
        CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, job_type TEXT NOT NULL DEFAULT 'slot_pipeline', title TEXT NOT NULL, params TEXT, status TEXT DEFAULT 'queued', current_stage TEXT DEFAULT 'Initializing', output_dir TEXT, error TEXT, created_at TEXT DEFAULT (datetime('now')), completed_at TEXT, parent_job_id TEXT, version INTEGER DEFAULT 1, FOREIGN KEY (user_id) REFERENCES users(id));
        CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_id);
    """)
    db.close()

# ── Migrate existing databases: add new columns if missing ──
def _migrate_db():
    try:
        db = get_db()
        cols = [r["name"] for r in db.execute("PRAGMA table_info(jobs)").fetchall()]
        if "parent_job_id" not in cols:
            db.execute("ALTER TABLE jobs ADD COLUMN parent_job_id TEXT")
        if "version" not in cols:
            db.execute("ALTER TABLE jobs ADD COLUMN version INTEGER DEFAULT 1")
        db.commit()
        db.close()
    except Exception:
        pass
init_db()
_migrate_db()

# ── Recover from crashes: check for orphaned "running" jobs from before restart ──
def _recover_stale_jobs():
    """On startup, check for jobs stuck in 'running'/'queued' from a previous crash.
    With start_new_session=True, workers may still be alive — only mark truly stale ones.
    Pipeline timeout is 90 min, so anything > 100 min is definitely stale."""
    try:
        db = get_db()
        stale = db.execute(
            "SELECT id, title FROM jobs WHERE status IN ('running','queued') "
            "AND created_at < datetime('now', '-100 minutes')"
        ).fetchall()
        for job in stale:
            db.execute(
                "UPDATE jobs SET status='failed', error='Timed out — exceeded maximum pipeline duration' WHERE id=?",
                (job["id"],)
            )
        if stale:
            db.commit()
            print(f"[RECOVERY] Marked {len(stale)} stale jobs as failed")
        db.close()
    except Exception:
        pass

_recover_stale_jobs()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session: return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated

@app.before_request
def _refresh_session():
    """Keep sessions alive for 30 days from last activity.
    This runs on every request and resets the 30-day expiry timer."""
    session.permanent = True

@app.before_request
def _csrf_origin_check():
    """Reject cross-origin POST/PUT/DELETE requests (poor-man's CSRF protection).
    Combined with SameSite=Lax cookies, this blocks most CSRF vectors."""
    if request.method in ("POST", "PUT", "DELETE"):
        origin = request.headers.get("Origin") or request.headers.get("Referer", "")
        if origin:
            from urllib.parse import urlparse
            allowed = request.host_url.rstrip("/")
            incoming = f"{urlparse(origin).scheme}://{urlparse(origin).netloc}"
            if incoming and incoming != allowed:
                return "Cross-origin request blocked", 403

def current_user(): return session.get("user", {})

BRAND_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Geist+Mono:wght@400;500&display=swap');
:root{
  --bg-void:#000000;--bg-surface:#0a0a0a;--bg-card:#111111;--bg-card-hover:#1a1a1a;--bg-input:#0a0a0a;
  --border:rgba(255,255,255,0.06);--border-hover:rgba(255,255,255,0.12);--border-focus:rgba(255,255,255,0.20);
  --text:#d4d4d4;--text-bright:#ffffff;--text-muted:#888888;--text-dim:#555555;
  --accent:#ffffff;--accent-soft:rgba(255,255,255,0.06);--accent-mid:rgba(255,255,255,0.10);--accent-bright:#ffffff;
  --success:#22c55e;--success-soft:rgba(34,197,94,0.08);--warning:#eab308;--warning-soft:rgba(234,179,8,0.08);--danger:#ef4444;--danger-soft:rgba(239,68,68,0.08);
  --radius:10px;--radius-lg:14px;--radius-xl:20px;
  --transition:all 0.15s ease;
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg-void);color:var(--text);min-height:100vh;-webkit-font-smoothing:antialiased;font-size:14px;line-height:1.6}
::selection{background:rgba(255,255,255,0.15);color:#fff}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.08);border-radius:10px}::-webkit-scrollbar-thumb:hover{background:rgba(255,255,255,0.14)}
a{color:var(--text-bright);text-decoration:none;transition:var(--transition)}a:hover{color:var(--text-muted)}

/* ── Layout Shell ── */
.topbar{position:sticky;top:0;z-index:100;display:flex;align-items:center;justify-content:space-between;padding:0 24px;height:52px;background:rgba(0,0,0,0.9);border-bottom:1px solid var(--border);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px)}
.logo{display:flex;align-items:center;gap:10px;font-weight:700;font-size:14px;letter-spacing:-0.03em;color:var(--text-bright);text-decoration:none}
.logo-mark{width:28px;height:28px;border-radius:8px;background:#fff;display:grid;place-items:center;font-size:13px;font-weight:800;color:#000}
.version-tag{font-size:10px;font-weight:500;color:var(--text-dim);font-family:'Geist Mono',monospace}
.user-pill{display:flex;align-items:center;gap:8px;padding:5px 14px 5px 5px;border-radius:24px;border:1px solid var(--border);font-size:12px;color:var(--text-muted);text-decoration:none;transition:var(--transition)}
.user-pill img{width:24px;height:24px;border-radius:50%}
.user-pill:hover{border-color:var(--border-hover);color:var(--text-bright)}

.shell{display:grid;grid-template-columns:220px 1fr;min-height:calc(100vh - 52px)}
.sidebar{padding:12px 0;border-right:1px solid var(--border);background:var(--bg-void);display:flex;flex-direction:column;gap:1px}
.sidebar a{display:flex;align-items:center;gap:10px;padding:9px 20px;font-size:13px;font-weight:400;color:var(--text-muted);text-decoration:none;transition:var(--transition);margin:0 8px;border-radius:8px}
.sidebar a:hover{color:var(--text-bright);background:var(--accent-soft)}
.sidebar a.active{color:var(--text-bright);background:var(--accent-soft);font-weight:500}
.sidebar a svg{width:16px;height:16px;opacity:0.4;flex-shrink:0}
.sidebar a:hover svg{opacity:0.65}.sidebar a.active svg{opacity:0.8}
.sidebar .section-label{font-size:10px;font-weight:500;letter-spacing:1.2px;color:var(--text-dim);padding:20px 20px 8px;text-transform:uppercase}

.main{padding:32px 48px;max-width:780px;width:100%;animation:fadeIn 0.15s ease}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
.page-title{font-size:22px;font-weight:700;color:var(--text-bright);margin-bottom:4px;letter-spacing:-0.03em}
.page-subtitle{color:var(--text-muted);font-size:13px;margin-bottom:28px;font-weight:400}

/* ── Cards — nearly invisible borders, float on black ── */
.card{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-lg);padding:24px;margin-bottom:14px;transition:var(--transition)}
.card:hover{border-color:var(--border-hover)}
.card h2{font-size:11px;font-weight:500;color:var(--text-muted);margin-bottom:16px;display:flex;align-items:center;gap:8px;letter-spacing:0.6px;text-transform:uppercase}

/* ── Forms ── */
label{display:block;font-size:12px;font-weight:500;color:var(--text-muted);margin-bottom:6px;letter-spacing:0.2px}
input,select,textarea{width:100%;padding:10px 14px;border-radius:var(--radius);border:1px solid var(--border);background:var(--bg-input);color:var(--text-bright);font-family:'Inter',sans-serif;font-size:13px;margin-bottom:16px;outline:none;transition:var(--transition)}
input:focus,select:focus,textarea:focus{border-color:var(--border-focus)}
input::placeholder,textarea::placeholder{color:var(--text-dim)}
textarea{min-height:70px;resize:vertical}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}

/* ── Buttons — flat, no gradients ── */
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;padding:9px 20px;border-radius:var(--radius);border:none;font-family:'Inter',sans-serif;font-size:13px;font-weight:500;cursor:pointer;transition:var(--transition);text-decoration:none}
.btn-primary{background:var(--text-bright);color:#000;font-weight:600}
.btn-primary:hover{opacity:0.85;color:#000}
.btn-primary:active{transform:scale(0.98)}
.btn-ghost{background:transparent;color:var(--text);border:1px solid var(--border)}
.btn-ghost:hover{border-color:var(--border-hover);color:var(--text-bright)}
.btn-sm{padding:6px 14px;font-size:12px;border-radius:8px}
.btn-full{width:100%}

/* ── Badges — minimal ── */
.badge{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:500;letter-spacing:0.1px}
.badge-running{color:var(--text-bright);background:var(--accent-soft)}
.badge-complete{color:var(--success)}
.badge-failed{color:var(--danger)}
.badge-queued{color:var(--warning)}
.badge-running::before{content:'';width:5px;height:5px;border-radius:50%;background:var(--text-bright);animation:pulse 1.8s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.15}}

/* ── History / List Items ── */
.history-item{display:grid;grid-template-columns:1fr 120px 140px 100px;align-items:center;padding:13px 20px;border-bottom:1px solid var(--border);font-size:13px;transition:var(--transition)}
.history-item:hover{background:var(--accent-soft)}
.history-title{font-weight:500;color:var(--text-bright)}
.history-type{color:var(--text-muted);font-size:12px;margin-top:2px}
.history-date{color:var(--text-dim);font-size:12px;font-family:'Geist Mono',monospace}
.history-actions{display:flex;gap:6px;justify-content:flex-end}

/* ── File Rows ── */
.file-row{display:flex;align-items:center;justify-content:space-between;padding:11px 20px;border-bottom:1px solid var(--border);font-size:13px;transition:var(--transition)}
.file-row:hover{background:var(--accent-soft)}
.file-row a{color:var(--text-bright);text-decoration:none;font-family:'Geist Mono',monospace;font-size:12px}
.file-row a:hover{color:var(--text-muted)}
.file-size{color:var(--text-dim);font-size:11px;font-family:'Geist Mono',monospace}

/* ── Stat Cards ── */
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:24px}
.stat-card{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);padding:16px;text-align:center;transition:var(--transition)}
.stat-card:hover{border-color:var(--border-hover)}
.stat-card .stat-icon{font-size:18px;margin-bottom:6px}
.stat-card .stat-val{font-size:16px;font-weight:600;color:var(--text-bright)}
.stat-card .stat-label{font-size:10px;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.6px;margin-top:4px;font-weight:500}
.stat-card.online .stat-val{color:var(--success)}
.stat-card.offline{opacity:0.35}

/* ── Feature Grid ── */
.feature-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px 10px}
.feature-grid label{display:flex;align-items:center;gap:8px;font-size:12px;font-weight:400;color:var(--text);text-transform:none;margin-bottom:0;cursor:pointer;padding:8px 10px;border-radius:8px;transition:var(--transition);border:1px solid transparent}
.feature-grid label:hover{background:var(--accent-soft);border-color:var(--border)}
.feature-grid label input{width:auto;margin:0;accent-color:#fff}
.feature-grid .feat-tag{font-size:9px;padding:2px 6px;border-radius:5px;font-weight:600;margin-left:auto;letter-spacing:0.2px}
.feat-tag.ip-risk{background:var(--danger-soft);color:var(--danger)}
.feat-tag.safe{background:var(--success-soft);color:var(--success)}
.feat-tag.banned{background:var(--warning-soft);color:var(--warning)}

/* ── Toggle / Options ── */
.toggle-section{padding:14px 18px;background:var(--bg-surface);border-radius:var(--radius);margin-top:12px;display:flex;flex-wrap:wrap;gap:18px;border:1px solid var(--border)}
.toggle-item{display:flex;align-items:center;gap:8px}
.toggle-item input{width:auto;margin:0;accent-color:#fff}
.toggle-item label{margin:0;font-size:12px;text-transform:none;color:var(--text-bright);font-weight:500}
.toggle-item .toggle-desc{font-size:11px;color:var(--text-dim)}

/* ── Login ── */
.login-wrap{min-height:100vh;display:grid;place-items:center;background:var(--bg-void)}
.login-box{text-align:center;padding:48px;width:380px;position:relative;z-index:1}
.login-box h1{font-size:24px;font-weight:700;letter-spacing:-0.03em;color:var(--text-bright);margin:24px 0 10px}
.login-box p{color:var(--text-dim);font-size:13px;margin-bottom:40px;line-height:1.7}
.google-btn{display:inline-flex;align-items:center;gap:10px;padding:12px 28px;border-radius:var(--radius);border:1px solid var(--border);background:transparent;color:var(--text-bright);font-family:'Inter',sans-serif;font-size:13px;font-weight:500;cursor:pointer;transition:var(--transition);text-decoration:none}
.google-btn:hover{border-color:var(--border-hover);background:var(--accent-soft)}
.google-btn svg{width:18px;height:18px}

/* ── Special Components ── */
.proto-frame{width:100%;height:600px;border:1px solid var(--border);border-radius:var(--radius);background:#000}
.audio-player{display:flex;align-items:center;gap:12px;padding:10px 20px;border-bottom:1px solid var(--border);font-size:13px}
.audio-player audio{height:32px;flex:1}
.audio-player .audio-name{font-family:'Geist Mono',monospace;font-size:11px;color:var(--text-bright);min-width:140px}

.cert-timeline{display:flex;gap:0;margin:16px 0}
.cert-step{flex:1;text-align:center;padding:12px 8px;position:relative}
.cert-step::after{content:'';position:absolute;top:26px;right:0;width:50%;height:2px;background:var(--border)}
.cert-step::before{content:'';position:absolute;top:26px;left:0;width:50%;height:2px;background:var(--border)}
.cert-step:first-child::before,.cert-step:last-child::after{display:none}
.cert-step .cert-dot{width:8px;height:8px;border-radius:50%;background:var(--text-bright);margin:0 auto 8px;position:relative;z-index:1}
.cert-step .cert-title{font-size:11px;font-weight:500;color:var(--text-bright)}
.cert-step .cert-sub{font-size:10px;color:var(--text-muted)}

.recon-input-group{display:flex;gap:12px;align-items:flex-end}
.recon-input-group input{margin-bottom:0;flex:1}
.recon-input-group .btn{white-space:nowrap;height:42px}
.empty-state{text-align:center;padding:48px 20px;color:var(--text-dim)}
.empty-state h3{font-size:14px;color:var(--text-muted);margin-bottom:6px;font-weight:500}
.empty-state p{font-size:13px}

/* ── Capability Grid ── */
.capability-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.cap-item{display:flex;align-items:center;gap:10px;padding:10px 14px;border-radius:8px;background:transparent;border:1px solid var(--border);font-size:12px;color:var(--text);transition:var(--transition)}
.cap-item:hover{border-color:var(--border-hover);background:var(--accent-soft)}
.cap-item b{color:var(--text-bright);font-weight:500}
.cap-item .cap-tag{font-size:10px;color:var(--text-dim);margin-left:auto;font-family:'Geist Mono',monospace}

/* ── Action Cards ── */
.action-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:24px}
.action-card{display:flex;align-items:center;gap:14px;padding:16px 20px;border-radius:var(--radius-lg);border:1px solid var(--border);background:transparent;text-decoration:none;transition:var(--transition)}
.action-card:hover{border-color:var(--border-hover);background:var(--accent-soft)}
.action-card .action-icon{font-size:20px;width:40px;height:40px;border-radius:10px;display:grid;place-items:center;background:var(--accent-soft);flex-shrink:0}
.action-card .action-text{font-size:13px;font-weight:600;color:var(--text-bright)}
.action-card .action-desc{font-size:12px;color:var(--text-dim);margin-top:2px}

/* ── Greeting ── */
.greeting{margin-bottom:24px}
.greeting h2{font-size:24px;font-weight:700;letter-spacing:-0.03em;color:var(--text-bright);margin-bottom:4px}
.greeting p{font-size:13px;color:var(--text-muted);font-weight:400}
.greeting .engine-tag{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:16px;border:1px solid var(--border);font-size:11px;color:var(--text-muted);font-weight:400;margin-top:8px}
.greeting .engine-tag::before{content:'';width:4px;height:4px;border-radius:50%;background:var(--success);animation:pulse 2s ease-in-out infinite}

/* ── Pipeline Form Sections ── */
.form-section{position:relative;counter-increment:form-step}
.form-section::before{content:counter(form-step);position:absolute;left:-36px;top:24px;width:24px;height:24px;border-radius:50%;background:var(--accent-soft);border:1px solid var(--border);display:grid;place-items:center;font-size:11px;font-weight:600;color:var(--text-muted);font-family:'Geist Mono',monospace}
.form-steps{counter-reset:form-step;padding-left:36px}

/* ── Log Terminal — Grok thinking style ── */
.log-terminal{background:var(--bg-surface);border:1px solid var(--border);border-radius:var(--radius-lg);padding:0;font-family:'Geist Mono',monospace;font-size:11.5px;line-height:1.8;height:calc(100vh - 200px);overflow-y:auto;white-space:pre-wrap;color:var(--text);position:relative}
.log-terminal .log-header{position:sticky;top:0;display:flex;align-items:center;gap:10px;padding:12px 16px;background:rgba(10,10,10,0.95);border-bottom:1px solid var(--border);backdrop-filter:blur(8px);z-index:10;font-size:12px;color:var(--text-dim)}
.log-terminal .log-body{padding:16px}

/* ── Shimmer Thinking Animation ── */
@keyframes shimmer-text{0%{background-position:-200% center}100%{background-position:200% center}}
.stage-shimmer{background:linear-gradient(90deg,var(--text-dim) 25%,var(--text-bright) 50%,var(--text-dim) 75%);background-size:200% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;animation:shimmer-text 2s linear infinite}

/* ── Progress Timeline ── */
.progress-timeline{display:flex;flex-direction:column;gap:0}
.progress-step{display:flex;align-items:flex-start;gap:12px;padding:8px 0;font-size:12px;color:var(--text-muted)}
.progress-step .step-dot{width:6px;height:6px;border-radius:50%;background:var(--text-dim);margin-top:5px;flex-shrink:0}
.progress-step.active .step-dot{background:var(--text-bright);box-shadow:0 0 6px rgba(255,255,255,0.3);animation:pulse 1.8s ease-in-out infinite}
.progress-step.done .step-dot{background:var(--success)}
.progress-step.done{color:var(--text-dim)}
.progress-step.active{color:var(--text-bright)}
.progress-step .step-time{font-family:'Geist Mono',monospace;color:var(--text-dim);font-size:11px;min-width:42px}

@media(max-width:768px){
  .shell{grid-template-columns:1fr}.sidebar{display:none}.main{padding:20px 16px;max-width:100%}
  .history-item{grid-template-columns:1fr 1fr;gap:8px}.stat-grid{grid-template-columns:repeat(2,1fr)}
  .feature-grid{grid-template-columns:1fr 1fr}.capability-grid{grid-template-columns:1fr}
  .action-grid{grid-template-columns:1fr}.greeting h2{font-size:20px}
  .form-steps{padding-left:0}.form-section::before{display:none}
}
"""

ICON_DASH = '<svg fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>'
ICON_PLUS = '<svg fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M12 5v14m7-7H5"/></svg>'
ICON_SEARCH = '<svg fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.35-4.35"/></svg>'
ICON_FOLDER = '<svg fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/></svg>'
ICON_CLOCK = '<svg fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>'
ICON_GLOBE = '<svg fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a15 15 0 014 9 15 15 0 01-4 9 15 15 0 01-4-9 15 15 0 014-9z"/></svg>'
ICON_DB = '<svg fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.66 3.58 3 8 3s8-1.34 8-3V5"/><path d="M4 12c0 1.66 3.58 3 8 3s8-1.34 8-3"/></svg>'
ICON_REVIEW = '<svg fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M9 12l2 2 4-4"/><circle cx="12" cy="12" r="9"/></svg>'
ICON_SETTINGS = '<svg fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M12 15a3 3 0 100-6 3 3 0 000 6z"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 01-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg>'
GOOGLE_SVG = '<svg viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 01-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>'
FAVICON_SVG = "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='8' fill='white'/><text x='16' y='22' text-anchor='middle' fill='black' font-size='18' font-weight='800'>A</text></svg>"

def layout(content, page="dashboard"):
    user = current_user()
    items = [("dashboard","Dashboard",ICON_DASH,"/"),("new","New Pipeline",ICON_PLUS,"/new"),("recon","State Recon",ICON_GLOBE,"/recon"),("reviews","Reviews",ICON_REVIEW,"/reviews"),("history","History",ICON_CLOCK,"/history"),("files","All Files",ICON_FOLDER,"/files"),("qdrant","Qdrant",ICON_DB,"/qdrant"),("settings","Settings",ICON_SETTINGS,"/settings")]
    nav = '<div class="section-label">Platform</div>'
    for k,l,i,h in items:
        nav += f'<a href="{h}" class="{"active" if page==k else ""}">{i} {l}</a>'
    pic = user.get("picture","")
    pic_tag = f'<img src="{_esc(pic)}" alt="" onerror="this.style.display=\'none\'" style="width:20px;height:20px;border-radius:50%">' if pic else ""
    name = user.get("name","User")
    return f'''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ARKAINBRAIN</title><link rel="icon" href="{FAVICON_SVG}"><style>{BRAND_CSS}</style></head><body>
<div class="topbar"><a href="/" class="logo"><div class="logo-mark">A</div>ARKAINBRAIN <span class="version-tag">v6</span></a><a href="/logout" class="user-pill">{pic_tag}{name} · Sign Out</a></div>
<div class="shell"><nav class="sidebar">{nav}<div class="section-label" style="margin-top:auto;padding-top:40px"><span style="color:var(--text-dim);font-size:10px;letter-spacing:0.5px">ArkainGames.com</span></div></nav><main class="main">{content}</main></div></body></html>'''

# ─── AUTH ───
@app.route("/login")
def login_page():
    return f'''<!DOCTYPE html><html><head><title>ARKAINBRAIN</title><link rel="icon" href="{FAVICON_SVG}"><style>{BRAND_CSS}</style></head><body>
<div class="login-wrap"><div class="login-box"><div class="logo-mark" style="width:44px;height:44px;font-size:20px;margin:0 auto;border-radius:12px">A</div><h1>ARKAINBRAIN</h1><p>AI-powered slot game intelligence.</p><a href="/auth/google" class="google-btn">{GOOGLE_SVG} Continue with Google</a><div style="margin-top:32px;font-size:11px;color:var(--text-dim)">ArkainGames.com · v5</div></div></div></body></html>'''

@app.route("/auth/google")
def google_login():
    return google.authorize_redirect(url_for("google_callback", _external=True))

@app.route("/auth/callback")
def google_callback():
    try:
        token = google.authorize_access_token()
        info = token.get("userinfo") or google.userinfo()
        db = get_db()
        db.execute("INSERT INTO users (id,email,name,picture) VALUES (?,?,?,?) ON CONFLICT(email) DO UPDATE SET name=excluded.name,picture=excluded.picture",
            (str(uuid.uuid4()), info["email"], info.get("name",""), info.get("picture","")))
        db.commit()
        row = db.execute("SELECT * FROM users WHERE email=?", (info["email"],)).fetchone()
        db.close()
        session.permanent = True  # 30-day session — survives browser close
        session["user"] = {"id":row["id"],"email":row["email"],"name":row["name"],"picture":row["picture"]}
        print(f"[AUTH] Login: {info['email']} → user_id={row['id']}")
        return redirect("/")
    except Exception as e:
        print(f"[AUTH] Error: {e}")
        return f"Auth error: {e}", 500

@app.route("/logout")
def logout():
    session.clear(); return redirect("/login")

# ─── DASHBOARD ───
@app.route("/")
@login_required
def dashboard():
    user = current_user()
    db = get_db()
    recent = db.execute("SELECT * FROM jobs WHERE user_id=? ORDER BY created_at DESC LIMIT 8", (user["id"],)).fetchall()
    db.close()
    rows = ""
    running_ids = []
    for job in recent:
        jid = job["id"]
        status = job["status"]
        stage = job["current_stage"] or ""
        bc = {"running":"badge-running","complete":"badge-complete","failed":"badge-failed"}.get(status,"badge-queued")
        tl = "Slot Pipeline" if job["job_type"]=="slot_pipeline" else "State Recon"
        dt = job["created_at"][:16].replace("T"," ") if job["created_at"] else ""
        stage_html = f'<span class="stage-shimmer" style="font-size:11px;margin-left:4px">{stage}</span>' if status == "running" and stage else ""
        act = f'<a href="/job/{jid}/files" class="btn btn-ghost btn-sm">Files</a>' if status=="complete" and job["output_dir"] else (f'<a href="/job/{jid}/logs" class="btn btn-ghost btn-sm" style="border-color:var(--border-hover);color:var(--text-bright)">Watch Live</a>' if status=="running" else "")
        rows += f'<div class="history-item" id="job-{jid}"><div><div class="history-title">{_esc(job["title"])}</div><div class="history-type">{tl}</div></div><div><span class="badge {bc}" id="badge-{jid}">{status}</span>{stage_html}</div><div class="history-date">{dt}</div><div class="history-actions" id="act-{jid}">{act}</div></div>'
        if status in ("running", "queued"):
            running_ids.append(jid)
    if not rows:
        rows = '<div class="empty-state"><h3>No pipelines yet</h3><p>Launch a Slot Pipeline or State Recon to get started.</p></div>'
    fname = user.get("name","").split()[0] if user.get("name") else "Operator"
    # Check for pending reviews
    review_banner = ""
    try:
        from tools.web_hitl import get_pending_reviews
        pending = get_pending_reviews()
        if pending:
            review_banner = f'<a href="/reviews" class="card" style="border-color:var(--border-hover);display:flex;align-items:center;gap:14px;text-decoration:none"><span class="badge badge-running" style="font-size:13px;padding:6px 14px">{len(pending)}</span><div><div style="font-weight:500;color:var(--text-bright);font-size:13px">Pipeline waiting for your review</div><div style="font-size:12px;color:var(--text-muted)">Click to approve, reject, or give feedback</div></div></a>'
    except Exception:
        pass

    # API status checks
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_serper = bool(os.getenv("SERPER_API_KEY"))
    has_elevenlabs = bool(os.getenv("ELEVENLABS_API_KEY"))
    has_qdrant = bool(os.getenv("QDRANT_URL"))

    api_cards = f'''<div class="stat-grid">
        <div class="stat-card {'online' if has_openai else 'offline'}"><div class="stat-icon">🧠</div><div class="stat-val">{'●' if has_openai else '○'}</div><div class="stat-label">OpenAI GPT-5</div></div>
        <div class="stat-card {'online' if has_serper else 'offline'}"><div class="stat-icon">🔍</div><div class="stat-val">{'●' if has_serper else '○'}</div><div class="stat-label">Serper Search</div></div>
        <div class="stat-card {'online' if has_elevenlabs else 'offline'}"><div class="stat-icon">🔊</div><div class="stat-val">{'●' if has_elevenlabs else '○'}</div><div class="stat-label">ElevenLabs</div></div>
        <div class="stat-card {'online' if has_qdrant else 'offline'}"><div class="stat-icon">🗃️</div><div class="stat-val">{'●' if has_qdrant else '○'}</div><div class="stat-label">Qdrant DB</div></div>
    </div>'''

    # Count totals
    db2 = get_db()
    total_jobs = db2.execute("SELECT COUNT(*) FROM jobs WHERE user_id=?", (user["id"],)).fetchone()[0]
    completed_jobs = db2.execute("SELECT COUNT(*) FROM jobs WHERE user_id=? AND status='complete'", (user["id"],)).fetchone()[0]
    db2.close()

    return layout(f'''
    <div class="greeting">
        <h2>Welcome back, {fname}</h2>
        <p>What would you like to build today?</p>
        <div class="engine-tag">GPT-5 · 6 Agents · OODA Convergence</div>
    </div>
    {review_banner}
    {api_cards}
    <div class="action-grid">
        <a href="/new" class="action-card"><div class="action-icon">🎰</div><div><div class="action-text">New Slot Pipeline</div><div class="action-desc">Concept → certified game package</div></div></a>
        <a href="/recon" class="action-card"><div class="action-icon">🌐</div><div><div class="action-text">State Recon</div><div class="action-desc">AI legal research for any jurisdiction</div></div></a>
    </div>
    <div class="card"><h2>Capabilities</h2>
        <div class="capability-grid">
            <div class="cap-item">🛰️ <b>Pre-Flight Intel</b> <span class="cap-tag">trend · jurisdiction</span></div>
            <div class="cap-item">🔬 <b>Vision QA</b> <span class="cap-tag">every image</span></div>
            <div class="cap-item">📐 <b>Math Optimizer</b> <span class="cap-tag">RTP ±0.1%</span></div>
            <div class="cap-item">🎭 <b>Agent Debate</b> <span class="cap-tag">OODA loop</span></div>
            <div class="cap-item">👤 <b>Player Behavior</b> <span class="cap-tag">5K sessions</span></div>
            <div class="cap-item">🔒 <b>Patent Scanner</b> <span class="cap-tag">IP check</span></div>
            <div class="cap-item">🎮 <b>HTML5 Prototype</b> <span class="cap-tag">playable</span></div>
            <div class="cap-item" style="{'opacity:0.35' if not has_elevenlabs else ''}">{'🔊' if has_elevenlabs else '🔇'} <b>Sound Design</b> <span class="cap-tag">{'on' if has_elevenlabs else '<a href=/settings style=color:var(--danger)>setup</a>'}</span></div>
            <div class="cap-item">📋 <b>Cert Planner</b> <span class="cap-tag">lab · cost</span></div>
            <div class="cap-item">⚔️ <b>Adversarial QA</b> <span class="cap-tag">devil's advocate</span></div>
        </div>
    </div>
    <div class="card" style="padding:0;overflow:hidden"><div style="padding:16px 20px 8px"><h2 style="margin-bottom:0">Recent Activity</h2></div>{rows}</div>
    <script>
    // Auto-refresh running jobs every 4 seconds
    const runningIds = {json.dumps(running_ids)};
    if (runningIds.length > 0) {{
        const poll = setInterval(() => {{
            let remaining = 0;
            runningIds.forEach(jid => {{
                fetch('/api/status/' + jid).then(r => r.json()).then(d => {{
                    const badge = document.getElementById('badge-' + jid);
                    if (!badge) return;
                    if (d.status !== badge.textContent) {{
                        badge.textContent = d.status;
                        badge.className = 'badge badge-' + (d.status === 'complete' ? 'complete' : d.status === 'failed' ? 'failed' : d.status === 'running' ? 'running' : 'queued');
                        if (d.status === 'complete' || d.status === 'failed') {{
                            setTimeout(() => location.reload(), 1000);
                        }}
                    }}
                    if (d.status === 'running' || d.status === 'queued') remaining++;
                }}).catch(() => {{}});
            }});
            if (remaining === 0) clearInterval(poll);
        }}, 4000);
    }}
    </script>''', "dashboard")

# ─── NEW PIPELINE ───
@app.route("/new")
@login_required
def new_pipeline():
    has_elevenlabs = bool(os.getenv("ELEVENLABS_API_KEY"))
    el_note = "" if has_elevenlabs else ' <span class="feat-tag ip-risk">No API key</span>'
    return layout(f'''
    <div class="greeting" style="margin-bottom:20px">
        <h2 style="font-size:20px">New Slot Pipeline</h2>
        <p>Describe your concept. Six agents research, design, model, illustrate, and certify it.</p>
    </div>
    <form action="/api/pipeline" method="POST">
    <div class="card"><h2>🎰 Game Concept</h2>
    <label>Theme / Concept</label><input name="theme" placeholder="e.g. Ancient Egyptian curse with escalating darkness" required style="font-size:15px;padding:14px 16px">
    <div class="row2"><div><label>Target Jurisdictions</label><input name="target_markets" placeholder="e.g. Georgia, Texas, UK, Malta" value="Georgia, Texas">
    <p style="font-size:10px;color:var(--text-muted);margin-top:-12px;margin-bottom:12px">US states, countries, or regulated markets. Auto-recon for unknown states.</p>
    </div>
    <div><label>Volatility</label><select name="volatility"><option value="low">Low</option><option value="medium" selected>Medium</option><option value="high">High</option><option value="very_high">Very High</option></select></div></div></div>

    <div class="card"><h2>📐 Math & Grid</h2>
    <div class="row3"><div><label>Target RTP %</label><input type="number" name="target_rtp" value="96.0" step="0.1" min="85" max="99"></div><div><label>Grid Cols</label><input type="number" name="grid_cols" value="5"></div><div><label>Grid Rows</label><input type="number" name="grid_rows" value="3"></div></div>
    <div class="row3"><div><label>Ways / Lines</label><input type="number" name="ways_or_lines" value="243"></div><div><label>Max Win Multiplier</label><input type="number" name="max_win_multiplier" value="5000"></div><div><label>Art Style</label><input name="art_style" value="Cinematic realism"></div></div></div>

    <div class="card"><h2>⚡ Features & Mechanics</h2>
    <div class="feature-grid">
        <label><input type="checkbox" name="features" value="free_spins" checked> Free Spins <span class="feat-tag safe">✓ Safe</span></label>
        <label><input type="checkbox" name="features" value="multipliers" checked> Multipliers <span class="feat-tag safe">✓ Safe</span></label>
        <label><input type="checkbox" name="features" value="expanding_wilds"> Expanding Wilds <span class="feat-tag safe">✓ Safe</span></label>
        <label><input type="checkbox" name="features" value="cascading_reels"> Cascading Reels <span class="feat-tag safe">Low IP</span></label>
        <label><input type="checkbox" name="features" value="mystery_symbols"> Mystery Symbols <span class="feat-tag safe">✓ Safe</span></label>
        <label><input type="checkbox" name="features" value="walking_wilds"> Walking Wilds <span class="feat-tag safe">Low IP</span></label>
        <label><input type="checkbox" name="features" value="cluster_pays"> Cluster Pays <span class="feat-tag safe">Low IP</span></label>
        <label><input type="checkbox" name="features" value="hold_and_spin"> Hold & Spin <span class="feat-tag ip-risk">Med IP</span></label>
        <label><input type="checkbox" name="features" value="bonus_buy"> Bonus Buy <span class="feat-tag banned">UK/SE ban</span></label>
        <label><input type="checkbox" name="features" value="progressive_jackpot"> Progressive Jackpot <span class="feat-tag ip-risk">+cost</span></label>
        <label><input type="checkbox" name="features" value="megaways"> Megaways™ <span class="feat-tag ip-risk">License req</span></label>
        <label><input type="checkbox" name="features" value="split_symbols"> Split Symbols <span class="feat-tag safe">Low IP</span></label>
    </div>
    <p style="font-size:10px;color:var(--text-muted);margin-top:12px">IP risk tags are pre-flight estimates. Patent Scanner verifies during execution.</p>
    <div style="margin-top:16px"><label>Competitor References</label><input name="competitor_references" placeholder="e.g. Book of Dead, Legacy of Dead, Sweet Bonanza">
    <label>Special Requirements</label><textarea name="special_requirements" placeholder="e.g. Must support mobile portrait mode, needs 5+ free spin retriggers, dark moody atmosphere..."></textarea></div></div>

    <div class="card"><h2>🤖 Pipeline Intelligence</h2>
    <div class="toggle-section">
        <div class="toggle-item"><input type="checkbox" name="enable_recon" value="on" checked id="recon"><label for="recon">🌐 Auto State Recon</label><span class="toggle-desc">Research unknown state laws</span></div>
        <div class="toggle-item"><input type="checkbox" name="enable_prototype" value="on" checked id="proto"><label for="proto">🎮 HTML5 Prototype</label><span class="toggle-desc">Playable demo</span></div>
        <div class="toggle-item"><input type="checkbox" name="enable_sound" value="on" {'checked' if has_elevenlabs else ''} id="snd"><label for="snd">🔊 Sound Design{el_note}</label><span class="toggle-desc">ElevenLabs SFX</span></div>
        <div class="toggle-item"><input type="checkbox" name="enable_cert_plan" value="on" checked id="cert"><label for="cert">📋 Cert Planning</label><span class="toggle-desc">Lab + timeline + cost</span></div>
        <div class="toggle-item"><input type="checkbox" name="enable_patent_scan" value="on" checked id="pat"><label for="pat">🔒 Patent/IP Scan</label><span class="toggle-desc">Mechanic conflicts</span></div>
    </div></div>

    <div class="card"><h2>⚙️ Execution Mode</h2>
    <div style="display:flex;gap:24px;align-items:center;margin-bottom:16px">
        <label style="display:flex;align-items:center;gap:8px;cursor:pointer;margin:0"><input type="radio" name="exec_mode" value="auto" checked style="width:auto;margin:0;accent-color:#fff" onchange="document.getElementById('variantOpts').style.display='none'"> <span style="text-transform:none;font-size:13px;color:var(--text-bright);font-weight:500">Auto</span><span style="font-size:11px;color:var(--text-dim);margin-left:4px">fully autonomous</span></label>
        <label style="display:flex;align-items:center;gap:8px;cursor:pointer;margin:0"><input type="radio" name="exec_mode" value="interactive" style="width:auto;margin:0;accent-color:#fff" onchange="document.getElementById('variantOpts').style.display='none'"> <span style="text-transform:none;font-size:13px;color:var(--text-bright);font-weight:500">Interactive</span><span style="font-size:11px;color:var(--text-dim);margin-left:4px">review at each stage</span></label>
        <label style="display:flex;align-items:center;gap:8px;cursor:pointer;margin:0"><input type="radio" name="exec_mode" value="variants" style="width:auto;margin:0;accent-color:#fff" onchange="document.getElementById('variantOpts').style.display='flex'"> <span style="text-transform:none;font-size:13px;color:var(--text-bright);font-weight:500">A/B Variants</span><span style="font-size:11px;color:var(--text-dim);margin-left:4px">2-5 parallel versions</span></label>
    </div>
    <div id="variantOpts" style="display:none;align-items:center;gap:12px;padding:12px;background:rgba(255,255,255,0.02);border:1px solid var(--border);border-radius:8px">
        <label style="font-size:12px;color:var(--text-muted);margin:0">Variants:</label>
        <select name="variant_count" style="width:60px;font-size:13px;padding:4px 8px;background:var(--bg-card);color:var(--text);border:1px solid var(--border);border-radius:6px">
            <option value="2">2</option><option value="3" selected>3</option><option value="4">4</option><option value="5">5</option></select>
        <span style="font-size:11px;color:var(--text-dim)">Conservative / Aggressive / Hybrid / Premium / Jackpot</span>
    </div></div>
    <button type="submit" id="launchBtn" class="btn btn-primary btn-full" style="padding:14px;font-size:14px;border-radius:var(--radius-lg)">Launch Pipeline &rarr;</button>
    <script>document.getElementById('launchBtn').addEventListener('click',function(e){{var m=document.querySelector('input[name=exec_mode]:checked').value;if(m==='variants'){{this.form.action='/api/variants'}}else{{if(m==='interactive'){{var h=document.createElement('input');h.type='hidden';h.name='interactive';h.value='on';this.form.appendChild(h)}}this.form.action='/api/pipeline'}}}});</script>
    </form>''', "new")

# ─── STATE RECON ───
@app.route("/recon")
@login_required
def recon_page():
    return layout(f'''
    <h2 class="page-title">{ICON_GLOBE} State Recon</h2>
    <p class="page-subtitle">Point at any US state. AI agents research laws, find loopholes, design compliant games.</p>
    <div class="card"><h2>{ICON_SEARCH} Research a State</h2><form action="/api/recon" method="POST"><label>US State Name</label><div class="recon-input-group"><input name="state" placeholder="e.g. North Carolina" required><button type="submit" class="btn btn-primary">Launch Recon</button></div></form></div>
    <div class="card"><h2>Pipeline Stages</h2><div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;text-align:center;padding:12px 0">
    <div><div style="font-size:22px;margin-bottom:6px">&#128269;</div><div style="font-size:12px;font-weight:600;color:var(--text-bright)">Legal Research</div><div style="font-size:11px;color:var(--text-dim)">Statutes, case law, AG opinions</div></div>
    <div><div style="font-size:22px;margin-bottom:6px">&#9878;&#65039;</div><div style="font-size:12px;font-weight:600;color:var(--text-bright)">Definition Analysis</div><div style="font-size:11px;color:var(--text-dim)">Element mapping, loophole ID</div></div>
    <div><div style="font-size:22px;margin-bottom:6px">&#127918;</div><div style="font-size:12px;font-weight:600;color:var(--text-bright)">Game Architecture</div><div style="font-size:11px;color:var(--text-dim)">Compliant mechanics design</div></div>
    <div><div style="font-size:22px;margin-bottom:6px">&#128203;</div><div style="font-size:12px;font-weight:600;color:var(--text-bright)">Defense Brief</div><div style="font-size:11px;color:var(--text-dim)">Courtroom-ready mapping</div></div></div></div>''', "recon")

# ─── HISTORY ───
@app.route("/history")
@login_required
def history_page():
    user = current_user()
    db = get_db()
    jobs = db.execute("SELECT * FROM jobs WHERE user_id=? ORDER BY created_at DESC LIMIT 50", (user["id"],)).fetchall()
    db.close()
    rows = ""
    for job in jobs:
        jid,status = job["id"], job["status"]
        bc = {"running":"badge-running","complete":"badge-complete","failed":"badge-failed"}.get(status,"badge-queued")
        tl = "Slot" if job["job_type"]=="slot_pipeline" else ("Recon" if job["job_type"]=="state_recon" else ("Iterate" if job["job_type"]=="iterate" else ("Variants" if job["job_type"]=="variant_parent" else ("Variant" if job["job_type"]=="variant" else job["job_type"]))))
        dt = job["created_at"][:16].replace("T"," ") if job["created_at"] else ""
        if job["job_type"] == "variant_parent":
            act = f'<a href="/job/{jid}/variants" class="btn btn-ghost btn-sm">Compare</a>' if status in ("running","complete") else ""
        elif status=="complete":
            act = f'<a href="/job/{jid}/files" class="btn btn-ghost btn-sm">Files</a>'
        elif status=="running":
            act = f'<a href="/job/{jid}/logs" class="btn btn-ghost btn-sm" style="border-color:var(--border-hover);color:var(--text-bright)">Watch Live</a>'
        else:
            act = ""
        err = f'<div style="font-size:11px;color:var(--danger);margin-top:2px">{job["error"][:80]}...</div>' if job["error"] else ""
        rows += f'<div class="history-item"><div><div class="history-title">{_esc(job["title"])}</div><div class="history-type">{tl}{err}</div></div><div><span class="badge {bc}">{status}</span></div><div class="history-date">{dt}</div><div class="history-actions">{act}</div></div>'
    if not rows: rows = '<div class="empty-state"><h3>No history yet</h3></div>'
    return layout(f'<h2 class="page-title" style="margin-bottom:24px">{ICON_CLOCK} Pipeline History</h2><div class="card" style="padding:0;overflow:hidden">{rows}</div>', "history")

# ─── FILES ───
@app.route("/files")
@login_required
def files_page():
    dirs = []
    if OUTPUT_DIR.exists():
        for d in sorted(OUTPUT_DIR.iterdir(), reverse=True):
            if d.is_dir():
                fc = sum(1 for _ in d.rglob("*") if _.is_file())
                ts = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                dirs.append({"name":d.name,"files":fc,"size":f"{ts/1024:.0f} KB" if ts<1048576 else f"{ts/1048576:.1f} MB","mtime":datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M")})
    rows = "".join(f'<div class="file-row"><a href="/files/{d["name"]}">{ICON_FOLDER} {d["name"]}</a><span class="file-size">{d["files"]} files &middot; {d["size"]}</span></div>' for d in dirs)
    if not rows: rows = '<div class="empty-state"><h3>No output files yet</h3></div>'
    return layout(f'<h2 class="page-title" style="margin-bottom:24px">{ICON_FOLDER} Output Files</h2><div class="card" style="padding:0;overflow:hidden">{rows}</div>', "files")

@app.route("/files/<path:subpath>")
@login_required
def browse_files(subpath):
    target = OUTPUT_DIR / subpath
    if not target.exists(): return "Not found", 404
    if target.is_file(): return send_from_directory(target.parent, target.name)
    files = [{"path":str(f.relative_to(target)),"url":f"/files/{f.relative_to(OUTPUT_DIR)}","size":f"{f.stat().st_size/1024:.1f} KB"} for f in sorted(target.rglob("*")) if f.is_file()]
    rows = "".join(f'<div class="file-row"><a href="{f["url"]}">{f["path"]}</a><span class="file-size">{f["size"]}</span></div>' for f in files)
    return layout(f'<div style="margin-bottom:20px"><a href="/files" style="color:var(--text-dim);font-size:12px;text-decoration:none">&larr; Back</a></div><h2 style="font-size:18px;font-weight:700;color:var(--text-bright);margin-bottom:16px">{subpath}</h2><div class="card" style="padding:0;overflow:hidden">{rows}</div>', "files")

# ─── JOB FILES ───
@app.route("/job/<job_id>/files")
@login_required
def job_files(job_id):
    db = get_db(); job = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone(); db.close()
    if not job or not job["output_dir"]: return "Not found", 404
    op = Path(job["output_dir"])
    if not op.exists(): return layout('<div class="card"><p style="color:var(--text-muted)">Output no longer exists.</p></div>')

    # Collect all files
    all_files = sorted(op.rglob("*"))
    files = [{"path":str(f.relative_to(op)),"url":f"/job/{job_id}/dl/{f.relative_to(op)}","size":f"{f.stat().st_size/1024:.1f} KB","ext":f.suffix.lower()} for f in all_files if f.is_file()]

    # Prototype section
    proto_html = ""
    proto_files = [f for f in files if f["path"].startswith("07_prototype") and f["ext"] == ".html"]
    if proto_files:
        proto_html = f'''<div class="card"><h2>🎮 Playable Prototype</h2>
            <iframe src="{proto_files[0]['url']}" class="proto-frame" title="Game Prototype"></iframe>
            <div style="margin-top:8px;text-align:center"><a href="{proto_files[0]['url']}" target="_blank" class="btn btn-ghost btn-sm">Open in new tab ↗</a></div></div>'''

    # Audio section
    audio_html = ""
    audio_files = [f for f in files if f["path"].startswith("04_audio") and f["ext"] in (".mp3", ".wav")]
    if audio_files:
        audio_rows = ""
        for af in audio_files:
            name = Path(af["path"]).stem
            audio_rows += f'<div class="audio-player"><span class="audio-name">{name}</span><audio controls preload="none" src="{af["url"]}"></audio><span class="file-size">{af["size"]}</span></div>'
        audio_html = f'<div class="card"><h2>🔊 AI Sound Design ({len(audio_files)} sounds)</h2><div style="max-height:400px;overflow-y:auto">{audio_rows}</div></div>'

    # Cert plan section
    cert_html = ""
    cert_file = op / "05_legal" / "certification_plan.json"
    if cert_file.exists():
        try:
            cert = json.loads(cert_file.read_text())
            markets = list(cert.get("per_market", {}).keys())
            timeline = cert.get("total_timeline", {})
            cost = cert.get("total_cost", {})
            lab = cert.get("recommended_lab", {})
            flags = cert.get("critical_flags", [])

            flags_html = "".join(f'<div style="padding:6px 10px;background:#ef444415;border-radius:6px;font-size:12px;color:var(--danger);margin-bottom:4px">⚠️ {fl}</div>' for fl in flags)

            cert_html = f'''<div class="card"><h2>📋 Certification Plan</h2>
                <div class="row3" style="margin-bottom:16px">
                    <div><label>Recommended Lab</label><div style="font-size:16px;font-weight:600;color:var(--text-bright)">{lab.get("name","TBD")}</div><div style="font-size:11px;color:var(--text-muted)">Covers {lab.get("covers_markets",0)}/{len(markets)} markets</div></div>
                    <div><label>Timeline (Parallel)</label><div style="font-size:16px;font-weight:700;color:var(--text-bright)">{timeline.get("parallel_testing_weeks","?")} weeks</div><div style="font-size:11px;color:var(--text-muted)">vs {timeline.get("sequential_testing_weeks","?")}w sequential</div></div>
                    <div><label>Total Cost Estimate</label><div style="font-size:16px;font-weight:700;color:var(--warning)">{cost.get("estimated_range","TBD")}</div></div>
                </div>
                {flags_html}
                <div style="margin-top:12px"><a href="/job/{job_id}/dl/05_legal/certification_plan.json" class="btn btn-ghost btn-sm">Download full plan JSON ↓</a></div></div>'''
        except Exception:
            pass

    # Patent scan section
    patent_html = ""
    patent_file = op / "00_preflight" / "patent_scan.json"
    if patent_file.exists():
        try:
            pscan = json.loads(patent_file.read_text())
            risk = pscan.get("risk_assessment", {})
            risk_level = risk.get("overall_ip_risk", "UNKNOWN")
            risk_color = {"HIGH":"var(--danger)","MEDIUM":"var(--warning)","LOW":"var(--success)"}.get(risk_level, "var(--text-muted)")
            hits = pscan.get("known_patent_hits", [])
            hits_rows = []
            for h in hits:
                risk_str = h.get("risk", "")
                rc = "var(--danger)" if risk_str.startswith("HIGH") else ("var(--warning)" if "MEDIUM" in risk_str else "var(--text-muted)")
                hits_rows.append(f'<div style="padding:6px 10px;background:var(--bg-input);border-radius:6px;font-size:12px;margin-bottom:4px"><b>{h.get("mechanic","")}</b> — {h.get("holder","")} <span style="color:{rc}">({risk_str})</span></div>')
            hits_html = "".join(hits_rows)

            patent_html = f'''<div class="card"><h2>🔒 Patent/IP Scan</h2>
                <div style="margin-bottom:12px"><span style="font-size:16px;font-weight:700;color:{risk_color}">{risk_level} RISK</span>
                <span style="font-size:12px;color:var(--text-muted);margin-left:8px">{risk.get("patent_conflicts",0)} conflicts, {risk.get("trademark_similar_names",0)} trademark matches</span></div>
                {hits_html if hits_html else '<div style="font-size:12px;color:var(--success)">No known patent conflicts detected.</div>'}
            </div>'''
        except Exception:
            pass

    # Revenue projection card (Phase 5)
    revenue_html = ""
    rev_file = op / "08_revenue" / "revenue_projection.json"
    if rev_file.exists():
        try:
            rev = json.loads(rev_file.read_text())
            ggr_365 = rev.get("ggr_365d", 0)
            ggr_90 = rev.get("ggr_90d", 0)
            arpdau = rev.get("arpdau", 0)
            be_days = rev.get("break_even_days", "?")
            roi = rev.get("roi_365d", 0)
            hold = rev.get("hold_pct", 0)
            cannibal = rev.get("cannibalization_risk", "?")
            cannibal_c = {"low":"var(--success)","medium":"var(--warning)","high":"var(--danger)"}.get(cannibal, "var(--text-muted)")
            roi_c = "var(--success)" if roi > 0 else "var(--danger)"

            # Mini monthly chart using CSS bars
            monthly = rev.get("ggr_monthly", [])
            max_ggr = max((m.get("ggr", 0) for m in monthly), default=1) or 1
            bars = ""
            for m in monthly[:12]:
                pct = min(100, int(m.get("ggr", 0) / max_ggr * 100))
                bars += f'<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:2px"><div style="width:100%;height:{pct}px;max-height:60px;background:linear-gradient(to top,rgba(255,255,255,0.05),rgba(255,255,255,0.15));border-radius:3px 3px 0 0"></div><span style="font-size:9px;color:var(--text-dim)">{m.get("month","")}</span></div>'

            # Market breakdown (top 3)
            mkt_rows = ""
            for mk in rev.get("market_breakdown", [])[:3]:
                mkt_rows += f'<div style="display:flex;justify-content:space-between;padding:4px 0;font-size:12px"><span style="color:var(--text-muted)">{mk.get("market","").upper()}</span><span style="color:var(--text-bright);font-family:var(--mono)">${mk.get("ggr_365d",0):,.0f}</span></div>'

            revenue_html = f'''<div class="card"><h2>&#128176; Revenue Projection</h2>
                <div class="row3" style="margin-bottom:16px">
                    <div><label>Annual GGR (365d)</label><div style="font-size:20px;font-weight:700;color:var(--text-bright)">${ggr_365:,.0f}</div></div>
                    <div><label>ARPDAU</label><div style="font-size:20px;font-weight:700;color:var(--text-bright)">${arpdau:.2f}</div></div>
                    <div><label>Hold %</label><div style="font-size:20px;font-weight:700;color:var(--text-bright)">{hold}%</div></div>
                </div>
                <div class="row3" style="margin-bottom:16px">
                    <div><label>Break-Even</label><div style="font-size:16px;font-weight:600;color:var(--warning)">{be_days} days</div></div>
                    <div><label>1-Year ROI</label><div style="font-size:16px;font-weight:600;color:{roi_c}">{roi:+.1f}%</div></div>
                    <div><label>Cannibalization</label><div style="font-size:16px;font-weight:600;color:{cannibal_c}">{cannibal.upper()}</div></div>
                </div>
                <div style="margin-bottom:16px"><label style="margin-bottom:8px;display:block">Monthly GGR Projection</label>
                    <div style="display:flex;gap:2px;align-items:flex-end;height:75px;padding:8px 0">{bars}</div></div>
                <div style="margin-bottom:12px"><label style="margin-bottom:6px;display:block">Top Markets</label>{mkt_rows}</div>
                <a href="/job/{job_id}/revenue" class="btn btn-ghost btn-sm" style="margin-top:4px">View full dashboard &rarr;</a></div>'''
        except Exception:
            pass

    # Engine export card (Phase 6)
    export_html = ""
    export_dir = op / "09_export" if op else None
    has_exports = export_dir and export_dir.exists() and any(export_dir.glob("*.zip"))
    if job["status"] == "complete":
        existing_zips = ""
        if has_exports:
            for zf in sorted(export_dir.glob("*.zip")):
                size_kb = zf.stat().st_size / 1024
                label = "Unity" if "unity" in zf.name else ("Godot" if "godot" in zf.name else "Generic")
                icon = {"Unity": "&#9898;", "Godot": "&#128430;", "Generic": "&#128230;"}.get(label, "&#128230;")
                existing_zips += f'<a href="/job/{job_id}/dl/09_export/{zf.name}" class="btn btn-ghost btn-sm" style="margin-right:8px;margin-bottom:6px">{icon} {label} ({size_kb:.0f} KB) &darr;</a>'

        export_html = f'''<div class="card"><h2>&#127918; Engine Export</h2>
            <p style="font-size:12px;color:var(--text-muted);margin-bottom:12px">Download engine-ready asset packages with structured data, sprites, audio, and auto-generated code.</p>
            <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px">
                <a href="/api/job/{job_id}/export?format=unity" class="btn btn-primary" style="font-size:12px;padding:8px 16px">&#9898; Unity Package</a>
                <a href="/api/job/{job_id}/export?format=godot" class="btn btn-primary" style="font-size:12px;padding:8px 16px">&#128430; Godot Package</a>
                <a href="/api/job/{job_id}/export?format=generic" class="btn btn-ghost" style="font-size:12px;padding:8px 16px">&#128230; Generic JSON</a>
            </div>
            {"<div style='margin-top:8px'><label style='font-size:11px;margin-bottom:4px;display:block'>Cached exports:</label>" + existing_zips + "</div>" if existing_zips else ""}
            <p style="font-size:10px;color:var(--text-dim);margin-top:8px">Includes: config.json, paytable.json, reelstrips.json, features.json, sprites, audio, {"SlotConfig.cs" if True else ""} + README</p></div>'''

    # Regular file list
    rows = "".join(f'<div class="file-row"><a href="{f["url"]}">{f["path"]}</a><span class="file-size">{f["size"]}</span></div>' for f in files)

    # Iterate button (only for completed jobs)
    iterate_btn = ""
    if job["status"] == "complete":
        iterate_btn = f'<a href="/job/{job_id}/iterate" class="btn btn-primary" style="font-size:13px;padding:8px 20px;margin-left:12px">🔄 Iterate</a>'
    # Variants button for variant_parent or variant jobs
    variants_btn = ""
    parent_for_variants = _rget(job, "parent_job_id") or job_id
    db_v = get_db()
    has_variants = db_v.execute("SELECT COUNT(*) as c FROM jobs WHERE parent_job_id=? AND job_type='variant'", (parent_for_variants,)).fetchone()["c"]
    db_v.close()
    if has_variants > 0:
        variants_btn = f'<a href="/job/{parent_for_variants}/variants" class="btn btn-ghost" style="font-size:13px;padding:8px 20px;margin-left:8px">🔀 Variants ({has_variants})</a>'

    # Version history + compare selector
    version_html = ""
    compare_html = ""
    db2 = get_db()
    root_id = _rget(job, "parent_job_id") or job_id
    versions = db2.execute("SELECT id,version,status,created_at FROM jobs WHERE id=? OR parent_job_id=? OR id=? ORDER BY version", (root_id, root_id, job_id)).fetchall()
    db2.close()
    if len(versions) > 1:
        vrows = ""
        compare_opts = ""
        for v in versions:
            active = " style='color:var(--text-bright);font-weight:600'" if v["id"] == job_id else ""
            sc = {"complete":"var(--success)","running":"var(--warning)","failed":"var(--danger)"}.get(v["status"],"var(--text-dim)")
            vrows += f'<a href="/job/{v["id"]}/files"{active}>v{v["version"] or 1} <span style="color:{sc};font-size:11px">{v["status"]}</span></a> '
            if v["id"] != job_id and v["status"] == "complete":
                compare_opts += f'<option value="{v["id"]}">v{v["version"] or 1}</option>'
        version_html = f'<div style="margin-bottom:12px;font-size:12px;color:var(--text-muted)">Versions: {vrows}</div>'
        if compare_opts:
            compare_html = f'''<div style="display:inline-flex;align-items:center;gap:6px;margin-left:12px">
                <select id="cmpSel" style="font-size:11px;padding:4px 8px;background:var(--bg-card);color:var(--text);border:1px solid var(--border);border-radius:6px">{compare_opts}</select>
                <button onclick="location.href='/job/{job_id}/diff/'+document.getElementById('cmpSel').value" class="btn btn-ghost" style="font-size:11px;padding:4px 12px">Compare ↔</button></div>'''

    return layout(f'''<div style="margin-bottom:20px"><a href="/history" style="color:var(--text-dim);font-size:12px;text-decoration:none">&larr; Back to History</a></div>
    <div style="display:flex;align-items:center;margin-bottom:4px"><h2 style="font-size:18px;font-weight:700;color:var(--text-bright)">{_esc(job["title"])}</h2>{iterate_btn}{variants_btn}{compare_html}</div>
    <p style="color:var(--text-muted);font-size:12px;margin-bottom:4px">{len(files)} files generated · v{_rget(job, "version") or 1}</p>
    {version_html}
    {proto_html}{audio_html}{patent_html}{cert_html}{revenue_html}{export_html}
    <div class="card" style="padding:0;overflow:hidden"><div style="padding:16px 16px 8px"><h2>📁 All Files</h2></div>{rows}</div>''', "history")

@app.route("/job/<job_id>/dl/<path:fp>")
@login_required
def job_dl(job_id, fp):
    db = get_db(); job = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone(); db.close()
    if not job or not job["output_dir"]: return "Not found", 404
    return send_from_directory(Path(job["output_dir"]), fp)


# ─── ITERATE: Selective Re-Run + Parameter Tweaker (Phase 3A-3B) ───

@app.route("/job/<job_id>/iterate")
@login_required
def job_iterate(job_id):
    user = current_user()
    db = get_db()
    job = db.execute("SELECT * FROM jobs WHERE id=? AND user_id=?", (job_id, user["id"])).fetchone()
    db.close()
    if not job: return "Not found", 404
    if job["status"] != "complete": return redirect(f"/job/{job_id}/logs")

    params = json.loads(job["params"]) if job["params"] else {}
    op = Path(job["output_dir"]) if job["output_dir"] else None

    # Read current simulation results for before/after comparison
    sim_data = {}
    if op:
        sim_path = op / "03_math" / "simulation_results.json"
        if sim_path.exists():
            try: sim_data = json.loads(sim_path.read_text())
            except Exception: pass

    # Read GDD quality audit if exists
    gdd_grade = "—"
    gdd_path = op / "02_design" / "gdd.md" if op else None
    has_gdd = gdd_path and gdd_path.exists()

    # Read convergence history
    conv_data = {}
    if op:
        conv_path = op / "02_design" / "convergence_history.json"
        if conv_path.exists():
            try: conv_data = json.loads(conv_path.read_text())
            except Exception: pass

    # Version info
    root_id = _rget(job, "parent_job_id") or job_id
    db2 = get_db()
    current_version = _rget(job, "version") or 1
    version_count = db2.execute("SELECT COUNT(*) as cnt FROM jobs WHERE id=? OR parent_job_id=?", (root_id, root_id)).fetchone()["cnt"]
    db2.close()
    next_version = version_count + 1

    # Current params display
    cur_rtp = params.get("target_rtp", 96.0)
    cur_max_win = params.get("max_win_multiplier", 5000)
    cur_vol = params.get("volatility", "medium")
    cur_markets = params.get("target_markets", [])
    cur_features = params.get("requested_features", [])
    measured_rtp = sim_data.get("measured_rtp", "—")
    max_win_achieved = sim_data.get("max_win_achieved", "—")
    hit_freq = sim_data.get("hit_frequency_pct", sim_data.get("hit_frequency", "—"))
    vol_idx = sim_data.get("volatility_index", "—")

    # Markets available for multi-select
    all_markets = ["UK","Malta","Sweden","Ontario","New Jersey","Michigan","Pennsylvania","Curaçao","Isle of Man","Gibraltar","Georgia","Texas","North Carolina","Florida"]
    market_options = ""
    for m in all_markets:
        checked = "checked" if m.lower() in [x.lower() for x in cur_markets] else ""
        market_options += f'<label class="iter-check"><input type="checkbox" name="target_markets" value="{m}" {checked}><span>{m}</span></label>'

    # Feature options
    all_features = ["free_spins","multipliers","expanding_wilds","cascading_reels","hold_and_spin","bonus_buy","scatter_pays","jackpot_progressive","cluster_pays","megaways"]
    feature_options = ""
    for f in all_features:
        checked = "checked" if f in cur_features else ""
        label = f.replace("_"," ").title()
        feature_options += f'<label class="iter-check"><input type="checkbox" name="features" value="{f}" {checked}><span>{label}</span></label>'

    return layout(f'''
    <div style="margin-bottom:20px"><a href="/job/{job_id}/files" style="color:var(--text-dim);font-size:12px;text-decoration:none">&larr; Back to {_esc(job["title"])}</a></div>
    <h2 class="page-title" style="margin-bottom:4px">🔄 Iterate — {_esc(job["title"])}</h2>
    <p style="color:var(--text-muted);font-size:12px;margin-bottom:24px">v{current_version} → v{next_version} · Re-run selected stages with new parameters</p>

    <form method="POST" action="/api/iterate" id="iterateForm">
    <input type="hidden" name="parent_job_id" value="{root_id}">
    <input type="hidden" name="source_job_id" value="{job_id}">
    <input type="hidden" name="source_output_dir" value="{job['output_dir'] or ''}">
    <input type="hidden" name="theme" value="{params.get('theme','')}">
    <input type="hidden" name="art_style" value="{params.get('art_style','')}">
    <input type="hidden" name="grid_cols" value="{params.get('grid_cols',5)}">
    <input type="hidden" name="grid_rows" value="{params.get('grid_rows',3)}">
    <input type="hidden" name="ways_or_lines" value="{params.get('ways_or_lines','243')}">

    <!-- Current Results -->
    <div class="card" style="margin-bottom:16px">
        <h2 style="font-size:15px;font-weight:600;margin-bottom:12px">Current Results (v{current_version})</h2>
        <div class="row4">
            <div><label style="font-size:11px;color:var(--text-muted)">Measured RTP</label><div style="font-size:20px;font-weight:700;color:var(--text-bright)">{measured_rtp}{'%' if isinstance(measured_rtp,(int,float)) else ''}</div></div>
            <div><label style="font-size:11px;color:var(--text-muted)">Max Win Achieved</label><div style="font-size:20px;font-weight:700;color:var(--text-bright)">{max_win_achieved}{'x' if isinstance(max_win_achieved,(int,float)) else ''}</div></div>
            <div><label style="font-size:11px;color:var(--text-muted)">Hit Frequency</label><div style="font-size:20px;font-weight:700;color:var(--text-bright)">{hit_freq}{'%' if isinstance(hit_freq,(int,float)) else ''}</div></div>
            <div><label style="font-size:11px;color:var(--text-muted)">Volatility Index</label><div style="font-size:20px;font-weight:700;color:var(--text-bright)">{vol_idx}</div></div>
        </div>
    </div>

    <!-- Parameter Tweaker (Phase 3B) -->
    <div class="card" style="margin-bottom:16px">
        <h2 style="font-size:15px;font-weight:600;margin-bottom:16px">⚙️ Parameter Tweaker</h2>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">
            <div>
                <label style="font-size:12px;color:var(--text-muted);display:block;margin-bottom:4px">Target RTP</label>
                <div style="display:flex;align-items:center;gap:10px">
                    <input type="range" name="target_rtp" min="85" max="99" step="0.1" value="{cur_rtp}"
                           oninput="this.nextElementSibling.textContent=this.value+'%'"
                           style="flex:1;accent-color:var(--text-bright)">
                    <span style="font-family:var(--mono);font-size:14px;font-weight:600;color:var(--text-bright);min-width:50px">{cur_rtp}%</span>
                </div>
            </div>
            <div>
                <label style="font-size:12px;color:var(--text-muted);display:block;margin-bottom:4px">Max Win Multiplier</label>
                <div style="display:flex;align-items:center;gap:10px">
                    <input type="range" name="max_win_multiplier" min="1000" max="50000" step="500" value="{cur_max_win}"
                           oninput="this.nextElementSibling.textContent=this.value+'x'"
                           style="flex:1;accent-color:var(--text-bright)">
                    <span style="font-family:var(--mono);font-size:14px;font-weight:600;color:var(--text-bright);min-width:60px">{cur_max_win}x</span>
                </div>
            </div>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
            <div>
                <label style="font-size:12px;color:var(--text-muted);display:block;margin-bottom:4px">Volatility</label>
                <select name="volatility" class="input-field" style="height:38px">
                    <option value="low" {"selected" if cur_vol=="low" else ""}>Low</option>
                    <option value="medium" {"selected" if cur_vol=="medium" else ""}>Medium</option>
                    <option value="medium_high" {"selected" if cur_vol=="medium_high" else ""}>Medium-High</option>
                    <option value="high" {"selected" if cur_vol=="high" else ""}>High</option>
                    <option value="extreme" {"selected" if cur_vol=="extreme" else ""}>Extreme</option>
                </select>
            </div>
            <div>
                <label style="font-size:12px;color:var(--text-muted);display:block;margin-bottom:4px">Special Requirements</label>
                <input type="text" name="special_requirements" value="{params.get('special_requirements','')}" class="input-field" placeholder="e.g. reduce free spin frequency">
            </div>
        </div>
    </div>

    <!-- Target Markets -->
    <div class="card" style="margin-bottom:16px">
        <h2 style="font-size:15px;font-weight:600;margin-bottom:12px">🌍 Target Markets</h2>
        <div style="display:flex;flex-wrap:wrap;gap:6px">{market_options}</div>
    </div>

    <!-- Features -->
    <div class="card" style="margin-bottom:16px">
        <h2 style="font-size:15px;font-weight:600;margin-bottom:12px">🎰 Features</h2>
        <div style="display:flex;flex-wrap:wrap;gap:6px">{feature_options}</div>
    </div>

    <!-- Selective Re-Run (Phase 3A) -->
    <div class="card" style="margin-bottom:16px">
        <h2 style="font-size:15px;font-weight:600;margin-bottom:12px">🔄 What to Re-Run</h2>
        <p style="font-size:12px;color:var(--text-muted);margin-bottom:12px">Select which stages to regenerate. Unselected stages keep their current output.</p>
        <div style="display:grid;gap:8px">
            <label class="iter-stage"><input type="checkbox" name="rerun_stages" value="math" checked><div><span style="font-weight:600">Math Model</span><span style="font-size:11px;color:var(--text-muted);display:block">Re-run Monte Carlo simulation with new parameters. Generates new reel strips, paytable, and sim results.</span></div></label>
            <label class="iter-stage"><input type="checkbox" name="rerun_stages" value="gdd"><div><span style="font-weight:600">GDD Patch</span><span style="font-size:11px;color:var(--text-muted);display:block">Update affected GDD sections to match new parameters (RTP budget, feature specs, volatility description).</span></div></label>
            <label class="iter-stage"><input type="checkbox" name="rerun_stages" value="art"><div><span style="font-weight:600">Art Assets</span><span style="font-size:11px;color:var(--text-muted);display:block">Regenerate all symbol images, backgrounds, and logo. Keep everything else.</span></div></label>
            <label class="iter-stage"><input type="checkbox" name="rerun_stages" value="compliance"><div><span style="font-weight:600">Compliance Review</span><span style="font-size:11px;color:var(--text-muted);display:block">Re-check regulations for changed markets or parameters. Generates new compliance report.</span></div></label>
            <label class="iter-stage"><input type="checkbox" name="rerun_stages" value="convergence"><div><span style="font-weight:600">Convergence Loop</span><span style="font-size:11px;color:var(--text-muted);display:block">Run full OODA convergence check to validate GDD ↔ Math ↔ Compliance alignment.</span></div></label>
        </div>
    </div>

    <!-- Submit -->
    <div style="display:flex;justify-content:flex-end;gap:12px;margin-bottom:40px">
        <a href="/job/{job_id}/files" class="btn btn-ghost">Cancel</a>
        <button type="submit" class="btn btn-primary" style="padding:10px 32px">🚀 Launch v{next_version}</button>
    </div>
    </form>

    <style>
        .row4 {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px }}
        .iter-check {{ display:inline-flex; align-items:center; gap:4px; padding:4px 10px; border:1px solid var(--border); border-radius:6px; cursor:pointer; font-size:12px; transition:border-color .15s }}
        .iter-check:has(input:checked) {{ border-color:var(--text-bright); background:rgba(255,255,255,0.04) }}
        .iter-check input {{ accent-color:var(--text-bright) }}
        .iter-stage {{ display:flex; align-items:flex-start; gap:10px; padding:10px 14px; border:1px solid var(--border); border-radius:8px; cursor:pointer; transition:border-color .15s }}
        .iter-stage:has(input:checked) {{ border-color:var(--text-bright); background:rgba(255,255,255,0.03) }}
        .iter-stage input {{ margin-top:3px; accent-color:var(--text-bright) }}
        input[type="range"] {{ height:4px; background:var(--border); border-radius:2px; -webkit-appearance:none; appearance:none }}
        input[type="range"]::-webkit-slider-thumb {{ -webkit-appearance:none; width:16px; height:16px; border-radius:50%; background:var(--text-bright); cursor:pointer }}
        @media(max-width:768px) {{ .row4 {{ grid-template-columns:repeat(2,1fr) }} }}
    </style>
    ''', "history")


@app.route("/api/iterate", methods=["POST"])
@login_required
def api_iterate():
    user = current_user()
    parent_id = request.form["parent_job_id"]
    source_id = request.form["source_job_id"]
    source_output = request.form["source_output_dir"]

    # Get next version number
    db = get_db()
    version_count = db.execute("SELECT COUNT(*) as cnt FROM jobs WHERE id=? OR parent_job_id=?", (parent_id, parent_id)).fetchone()["cnt"]
    next_version = version_count + 1

    # Build iteration params
    params = {
        "theme": request.form["theme"],
        "target_markets": request.form.getlist("target_markets"),
        "volatility": request.form.get("volatility", "medium"),
        "target_rtp": float(request.form.get("target_rtp", 96)),
        "grid_cols": int(request.form.get("grid_cols", 5)),
        "grid_rows": int(request.form.get("grid_rows", 3)),
        "ways_or_lines": request.form.get("ways_or_lines", "243"),
        "max_win_multiplier": int(request.form.get("max_win_multiplier", 5000)),
        "art_style": request.form.get("art_style", "Cinematic realism"),
        "requested_features": request.form.getlist("features"),
        "special_requirements": request.form.get("special_requirements", ""),
    }

    iterate_config = {
        "source_job_id": source_id,
        "source_output_dir": source_output,
        "rerun_stages": request.form.getlist("rerun_stages"),
        "parent_job_id": parent_id,
        "version": next_version,
    }

    job_id = str(uuid.uuid4())[:8]
    db.execute(
        "INSERT INTO jobs (id, user_id, job_type, title, params, status, parent_job_id, version) VALUES (?,?,?,?,?,?,?,?)",
        (job_id, user["id"], "iterate", f"{params['theme']} v{next_version}",
         json.dumps({**params, "_iterate": iterate_config}), "queued", parent_id, next_version)
    )
    db.commit()
    db.close()

    _spawn_worker(job_id, "iterate", json.dumps({**params, "_iterate": iterate_config}))
    return redirect(f"/job/{job_id}/logs")


def _load_job_metrics(output_dir):
    """Load key metrics from a job output dir for comparison."""
    od = Path(output_dir) if output_dir else None
    data = {"rtp":"—","max_win":"—","hit_freq":"—","vol_idx":"—","gdd_words":0,"symbols":0,"compliance":"—","gdd_sections":[],"rtp_breakdown":{},"ggr_365d":"—","arpdau":"—","roi_365d":"—","break_even_days":"—"}
    if not od or not od.exists(): return data
    sim_path = od / "03_math" / "simulation_results.json"
    if sim_path.exists():
        try:
            sim = json.loads(sim_path.read_text())
            data["rtp"]=sim.get("measured_rtp","—"); data["max_win"]=sim.get("max_win_achieved","—")
            data["hit_freq"]=sim.get("hit_frequency_pct",sim.get("hit_frequency","—")); data["vol_idx"]=sim.get("volatility_index","—")
            data["rtp_breakdown"]=sim.get("rtp_breakdown",{})
        except Exception: pass
    pt_path = od / "03_math" / "paytable.csv"
    if pt_path.exists():
        try:
            import csv as _csv, io as _io
            data["symbols"] = max(0, sum(1 for _ in _csv.reader(_io.StringIO(pt_path.read_text()))) - 1)
        except Exception: pass
    gdd_path = od / "02_design" / "gdd.md"
    if gdd_path.exists():
        try:
            gdd_text = gdd_path.read_text(encoding="utf-8", errors="replace")
            data["gdd_words"] = len(gdd_text.split())
            import re as _re; data["gdd_sections"] = _re.findall(r'^## .+', gdd_text, _re.MULTILINE)
        except Exception: pass
    comp_path = od / "05_legal" / "compliance_report.json"
    if comp_path.exists():
        try: data["compliance"] = json.loads(comp_path.read_text()).get("overall_status","—")
        except Exception: pass
    rev_path = od / "08_revenue" / "revenue_projection.json"
    if rev_path.exists():
        try:
            rv = json.loads(rev_path.read_text())
            data["ggr_365d"] = rv.get("ggr_365d", "—")
            data["arpdau"] = rv.get("arpdau", "—")
            data["roi_365d"] = rv.get("roi_365d", "—")
            data["break_even_days"] = rv.get("break_even_days", "—")
        except Exception: pass
    return data


@app.route("/job/<job_id>/diff/<other_id>")
@login_required
def job_diff(job_id, other_id):
    user = current_user(); db = get_db()
    job_a = db.execute("SELECT * FROM jobs WHERE id=? AND user_id=?", (job_id, user["id"])).fetchone()
    job_b = db.execute("SELECT * FROM jobs WHERE id=? AND user_id=?", (other_id, user["id"])).fetchone()
    db.close()
    if not job_a or not job_b: return "Not found", 404
    a = _load_job_metrics(job_a["output_dir"]); b = _load_job_metrics(job_b["output_dir"])
    va = _rget(job_a, "version") or 1; vb = _rget(job_b, "version") or 1

    def _dc(label, val_a, val_b, fmt="", hib=None):
        sa = f"{val_a}{fmt}" if isinstance(val_a,(int,float)) else str(val_a)
        sb = f"{val_b}{fmt}" if isinstance(val_b,(int,float)) else str(val_b)
        delta = ""
        if isinstance(val_a,(int,float)) and isinstance(val_b,(int,float)):
            d = val_b - val_a; sign = "+" if d > 0 else ""
            color = "var(--text-muted)"
            if hib is True: color = "var(--success)" if d > 0 else ("var(--danger)" if d < 0 else color)
            elif hib is False: color = "var(--danger)" if d > 0 else ("var(--success)" if d < 0 else color)
            delta = f'<span style="font-size:11px;color:{color};margin-left:4px">{sign}{d:.2f}{fmt}</span>' if d != 0 else ""
        return f'<tr><td style="font-size:12px;color:var(--text-muted);padding:6px 0">{label}</td><td style="font-family:var(--mono);font-size:13px;padding:6px 12px">{sa}</td><td style="font-family:var(--mono);font-size:13px;font-weight:600;padding:6px 12px">{sb}{delta}</td></tr>'

    rows = _dc("Measured RTP",a["rtp"],b["rtp"],"%") + _dc("Max Win",a["max_win"],b["max_win"],"x") + _dc("Hit Frequency",a["hit_freq"],b["hit_freq"],"%",True) + _dc("Volatility Index",a["vol_idx"],b["vol_idx"],"") + _dc("Symbols",a["symbols"],b["symbols"],"") + _dc("GDD Words",a["gdd_words"],b["gdd_words"],"",True) + _dc("Compliance",a["compliance"],b["compliance"],"") + _dc("Annual GGR",a.get("ggr_365d","—"),b.get("ggr_365d","—"),"",True) + _dc("ARPDAU",a.get("arpdau","—"),b.get("arpdau","—"),"",True) + _dc("1Y ROI",a.get("roi_365d","—"),b.get("roi_365d","—"),"%",True) + _dc("Break-Even",a.get("break_even_days","—"),b.get("break_even_days","—")," days",False)

    rtp_a = a.get("rtp_breakdown",{}); rtp_b = b.get("rtp_breakdown",{})
    rtp_rows = "".join(_dc(k.replace("_"," ").title(), rtp_a.get(k,0), rtp_b.get(k,0), "%") for k in sorted(set(list(rtp_a)+list(rtp_b))) if isinstance(rtp_a.get(k,0),(int,float)) and isinstance(rtp_b.get(k,0),(int,float)))
    rtp_sec = f'<div class="card" style="margin-top:16px"><h2 style="font-size:15px;font-weight:600;margin-bottom:12px">RTP Breakdown</h2><table style="width:100%;border-collapse:collapse"><tr><th></th><th style="font-size:11px;color:var(--text-muted);text-align:left;padding:4px 12px">v{va}</th><th style="font-size:11px;color:var(--text-muted);text-align:left;padding:4px 12px">v{vb}</th></tr>{rtp_rows}</table></div>' if rtp_rows else ""

    secs_a = set(a.get("gdd_sections",[])); secs_b = set(b.get("gdd_sections",[]))
    gdd_diff = ""
    if (secs_b - secs_a) or (secs_a - secs_b):
        items = "".join(f'<div style="color:var(--success);font-size:12px">+ {s}</div>' for s in secs_b - secs_a)
        items += "".join(f'<div style="color:var(--danger);font-size:12px">- {s}</div>' for s in secs_a - secs_b)
        gdd_diff = f'<div class="card" style="margin-top:16px"><h2 style="font-size:15px;font-weight:600;margin-bottom:8px">GDD Section Changes</h2>{items}</div>'

    return layout(f'''<div style="margin-bottom:20px"><a href="/job/{job_id}/files" style="color:var(--text-dim);font-size:12px;text-decoration:none">&larr; Back to v{va}</a></div>
    <h2 class="page-title" style="margin-bottom:4px">&#8596; Version Diff</h2>
    <p style="color:var(--text-muted);font-size:12px;margin-bottom:24px">{_esc(job_a["title"])} — v{va} vs v{vb}</p>
    <div class="card"><h2 style="font-size:15px;font-weight:600;margin-bottom:12px">Key Metrics</h2>
        <table style="width:100%;border-collapse:collapse"><tr><th></th><th style="font-size:11px;color:var(--text-muted);text-align:left;padding:4px 12px">v{va}</th><th style="font-size:11px;color:var(--text-muted);text-align:left;padding:4px 12px">v{vb}</th></tr>{rows}</table></div>
    {rtp_sec}{gdd_diff}
    <div style="display:flex;gap:12px;margin-top:24px;margin-bottom:40px">
        <a href="/job/{job_id}/files" class="btn btn-ghost" style="flex:1;text-align:center">View v{va}</a>
        <a href="/job/{other_id}/files" class="btn btn-ghost" style="flex:1;text-align:center">View v{vb}</a></div>''', "history")


@app.route("/job/<job_id>/variants")
@login_required
def job_variants(job_id):
    user = current_user(); db = get_db()
    parent = db.execute("SELECT * FROM jobs WHERE id=? AND user_id=?", (job_id, user["id"])).fetchone()
    if not parent: db.close(); return "Not found", 404
    variants = db.execute("SELECT * FROM jobs WHERE parent_job_id=? AND job_type='variant' ORDER BY version", (job_id,)).fetchall()
    db.close()
    if not variants:
        return layout(f'<div class="card"><p style="color:var(--text-muted)">No variants yet.</p><a href="/history" class="btn btn-ghost" style="margin-top:12px">Back</a></div>', "history")

    variant_data = []
    for v in variants:
        m = _load_job_metrics(v["output_dir"]); params = json.loads(v["params"]) if v["params"] else {}
        vc = params.get("_variant", {})
        variant_data.append({"id":v["id"],"status":v["status"],"label":vc.get("label",f"V{_rget(v, 'version','?')}"),"strategy":vc.get("strategy",""),"metrics":m})

    header = '<th style="font-size:11px;color:var(--text-muted);padding:6px 12px;text-align:left">Metric</th>'
    for vd in variant_data:
        sc = {"complete":"var(--success)","running":"var(--warning)","failed":"var(--danger)"}.get(vd["status"],"var(--text-dim)")
        header += f'<th style="font-size:12px;padding:6px 12px;text-align:left"><span style="color:var(--text-bright);font-weight:600">{vd["label"]}</span><br><span style="font-size:10px;color:{sc}">{vd["status"]}</span></th>'

    def _vr(label,key,fmt=""):
        c = f'<td style="font-size:12px;color:var(--text-muted);padding:6px 12px">{label}</td>'
        for vd in variant_data:
            val = vd["metrics"].get(key,"—")
            c += f'<td style="font-family:var(--mono);font-size:13px;padding:6px 12px">{val}{fmt if isinstance(val,(int,float)) else ""}</td>'
        return f"<tr>{c}</tr>"
    trows = _vr("RTP","rtp","%")+_vr("Max Win","max_win","x")+_vr("Hit Freq","hit_freq","%")+_vr("Volatility","vol_idx","")+_vr("Symbols","symbols","")+_vr("GDD Words","gdd_words","")+_vr("Compliance","compliance","")+_vr("Annual GGR","ggr_365d","")+_vr("ARPDAU","arpdau","")+_vr("1Y ROI","roi_365d","%")+_vr("Break-Even","break_even_days"," days")

    strat = "".join(f'<div class="card" style="margin-bottom:8px"><div style="display:flex;justify-content:space-between;align-items:center"><div><h3 style="font-size:14px;font-weight:600;color:var(--text-bright);margin:0">{vd["label"]}</h3><p style="font-size:12px;color:var(--text-muted);margin:4px 0 0">{vd["strategy"]}</p></div><a href="/job/{vd["id"]}/files" class="btn btn-ghost" style="font-size:11px;padding:4px 12px">View &rarr;</a></div></div>' for vd in variant_data)

    return layout(f'''<div style="margin-bottom:20px"><a href="/history" style="color:var(--text-dim);font-size:12px;text-decoration:none">&larr; History</a></div>
    <h2 class="page-title" style="margin-bottom:4px">&#128256; Variant Comparison</h2>
    <p style="color:var(--text-muted);font-size:12px;margin-bottom:24px">{_esc(parent["title"])} — {len(variant_data)} variants</p>
    <div class="card" style="overflow-x:auto"><h2 style="font-size:15px;font-weight:600;margin-bottom:12px">Side-by-Side</h2>
        <table style="width:100%;border-collapse:collapse"><tr>{header}</tr>{trows}</table></div>
    <div style="margin-top:16px">{strat}</div>
    <div style="margin-top:24px;margin-bottom:40px"><a href="/history" class="btn btn-ghost">Back</a></div>''', "history")


@app.route("/api/variants", methods=["POST"])
@login_required
def api_launch_variants():
    user = current_user()
    variant_count = max(2, min(int(request.form.get("variant_count", 3)), 5))
    base_params = {"theme":request.form["theme"],"target_markets":[m.strip() for m in request.form.get("target_markets","Georgia, Texas").split(",")],"volatility":request.form.get("volatility","medium"),"target_rtp":float(request.form.get("target_rtp",96)),"grid_cols":int(request.form.get("grid_cols",5)),"grid_rows":int(request.form.get("grid_rows",3)),"ways_or_lines":request.form.get("ways_or_lines","243"),"max_win_multiplier":int(request.form.get("max_win_multiplier",5000)),"art_style":request.form.get("art_style","Cinematic realism"),"requested_features":request.form.getlist("features"),"competitor_references":[r.strip() for r in request.form.get("competitor_references","").split(",") if r.strip()],"special_requirements":request.form.get("special_requirements",""),"enable_recon":request.form.get("enable_recon")=="on"}

    parent_id = str(uuid.uuid4())[:8]; db = get_db()
    db.execute("INSERT INTO jobs (id,user_id,job_type,title,params,status,current_stage) VALUES (?,?,?,?,?,?,?)",
        (parent_id,user["id"],"variant_parent",f"{base_params['theme']} (variants)",json.dumps(base_params),"running",f"Spawning {variant_count} variants"))
    db.commit(); db.close()

    STRATEGIES = [
        {"label":"Conservative","strategy":"Lower volatility, proven features, safe theme. High hit freq, steady wins.","vol_adj":-1,"rtp_adj":0.5,"max_win_adj":-0.3},
        {"label":"Aggressive","strategy":"Higher volatility, novel mechanics, bold theme. Max win and feature drama.","vol_adj":1,"rtp_adj":-0.3,"max_win_adj":0.5},
        {"label":"Hybrid","strategy":"Mid-volatility with one novel mechanic. Balance excitement and session length.","vol_adj":0,"rtp_adj":0,"max_win_adj":0},
        {"label":"Premium","strategy":"High RTP, moderate volatility, polished. Target experienced fairness-seekers.","vol_adj":0,"rtp_adj":1.0,"max_win_adj":-0.2},
        {"label":"Jackpot Focus","strategy":"Low base RTP, high jackpot contribution. Dream-big psychology.","vol_adj":1,"rtp_adj":-0.5,"max_win_adj":1.0},
    ]
    VOL_LEVELS = ["low","medium","medium_high","high","extreme"]
    variant_ids = []
    for i in range(variant_count):
        s = STRATEGIES[i % len(STRATEGIES)]; vid = str(uuid.uuid4())[:8]; variant_ids.append(vid)
        vp = {**base_params}
        ci = VOL_LEVELS.index(vp["volatility"]) if vp["volatility"] in VOL_LEVELS else 1
        vp["volatility"] = VOL_LEVELS[max(0,min(len(VOL_LEVELS)-1,ci+s["vol_adj"]))]
        vp["target_rtp"] = round(max(85,min(99,vp["target_rtp"]+s["rtp_adj"])),1)
        vp["max_win_multiplier"] = max(1000,int(vp["max_win_multiplier"]*(1+s["max_win_adj"])))
        vp["special_requirements"] = f"VARIANT STRATEGY: {s['strategy']}\n{vp.get('special_requirements','')}"
        vp["_variant"] = {"label":s["label"],"strategy":s["strategy"],"variant_index":i+1}
        db2 = get_db()
        db2.execute("INSERT INTO jobs (id,user_id,job_type,title,params,status,parent_job_id,version) VALUES (?,?,?,?,?,?,?,?)",
            (vid,user["id"],"variant",f"{base_params['theme']} — {s['label']}",json.dumps(vp),"queued",parent_id,i+1))
        db2.commit(); db2.close()
        _spawn_worker(vid, "pipeline", json.dumps(vp))

    db3 = get_db()
    db3.execute("UPDATE jobs SET params=?,current_stage=? WHERE id=?",
        (json.dumps({**base_params,"_variant_ids":variant_ids}),f"{variant_count} variants running",parent_id))
    db3.commit(); db3.close()
    return redirect(f"/job/{parent_id}/variants")


# ─── REVENUE DASHBOARD (Phase 5B) ───

@app.route("/job/<job_id>/revenue")
@login_required
def job_revenue(job_id):
    user = current_user(); db = get_db()
    job = db.execute("SELECT * FROM jobs WHERE id=? AND user_id=?", (job_id, user["id"])).fetchone()
    db.close()
    if not job: return "Not found", 404
    op = Path(job["output_dir"]) if job["output_dir"] else None
    rev_file = op / "08_revenue" / "revenue_projection.json" if op else None
    if not rev_file or not rev_file.exists():
        return layout(f'<div class="card"><p style="color:var(--text-muted)">No revenue projection available for this job.</p><a href="/job/{job_id}/files" class="btn btn-ghost" style="margin-top:12px">Back</a></div>', "history")

    try:
        rev = json.loads(rev_file.read_text())
    except (json.JSONDecodeError, ValueError, OSError):
        return layout(f'<div class="card"><p style="color:var(--text-muted)">Revenue data is corrupted. Re-run the pipeline to regenerate.</p><a href="/job/{job_id}/files" class="btn btn-ghost" style="margin-top:12px">Back</a></div>', "history")

    # ── Hero metrics ──
    hero = f'''<div class="row3" style="margin-bottom:24px">
        <div class="stat-card"><div class="stat-val" style="font-size:24px">${rev.get("ggr_365d",0):,.0f}</div><div class="stat-label">Annual GGR (365d)</div></div>
        <div class="stat-card"><div class="stat-val" style="font-size:24px">${rev.get("arpdau",0):.2f}</div><div class="stat-label">ARPDAU</div></div>
        <div class="stat-card"><div class="stat-val" style="font-size:24px">{rev.get("hold_pct",0)}%</div><div class="stat-label">Effective Hold</div></div>
    </div>
    <div class="row3" style="margin-bottom:24px">
        <div class="stat-card"><div class="stat-val" style="font-size:20px">{rev.get("break_even_days","?")} days</div><div class="stat-label">Break-Even</div></div>
        <div class="stat-card"><div class="stat-val" style="font-size:20px;color:{"var(--success)" if rev.get("roi_365d",0)>0 else "var(--danger)"}">{rev.get("roi_365d",0):+.1f}%</div><div class="stat-label">1-Year ROI</div></div>
        <div class="stat-card"><div class="stat-val" style="font-size:20px">{rev.get("daily_active_users",0):,}</div><div class="stat-label">Projected DAU</div></div>
    </div>'''

    # ── GGR Period Cards ──
    periods = f'''<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px">
        <div style="padding:16px;background:var(--bg-card);border:1px solid var(--border);border-radius:8px;text-align:center">
            <div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">30-Day</div>
            <div style="font-size:18px;font-weight:700;color:var(--text-bright)">${rev.get("ggr_30d",0):,.0f}</div></div>
        <div style="padding:16px;background:var(--bg-card);border:1px solid var(--border);border-radius:8px;text-align:center">
            <div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">90-Day</div>
            <div style="font-size:18px;font-weight:700;color:var(--text-bright)">${rev.get("ggr_90d",0):,.0f}</div></div>
        <div style="padding:16px;background:var(--bg-card);border:1px solid var(--border);border-radius:8px;text-align:center">
            <div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">180-Day</div>
            <div style="font-size:18px;font-weight:700;color:var(--text-bright)">${rev.get("ggr_180d",0):,.0f}</div></div>
        <div style="padding:16px;background:var(--bg-card);border:1px solid var(--border);border-radius:8px;text-align:center">
            <div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">365-Day</div>
            <div style="font-size:18px;font-weight:700;color:var(--text-bright)">${rev.get("ggr_365d",0):,.0f}</div></div>
    </div>'''

    # ── Monthly GGR Chart (CSS bar chart) ──
    monthly = rev.get("ggr_monthly", [])
    max_ggr = max((m.get("ggr", 0) for m in monthly), default=1) or 1
    bars = ""
    for m in monthly[:12]:
        pct = min(100, int(m.get("ggr", 0) / max_ggr * 100))
        ggr_val = m.get("ggr", 0)
        bars += f'''<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:4px">
            <span style="font-size:9px;color:var(--text-dim);font-family:var(--mono)">${ggr_val:,.0f}</span>
            <div style="width:100%;height:{max(4, pct)}px;max-height:80px;background:linear-gradient(to top,rgba(255,255,255,0.08),rgba(255,255,255,0.2));border-radius:4px 4px 0 0"></div>
            <span style="font-size:10px;color:var(--text-muted)">M{m.get("month","")}</span>
            <span style="font-size:9px;color:var(--text-dim)">{m.get("dau",0):,} DAU</span></div>'''
    chart = f'''<div class="card"><h2 style="font-size:15px;font-weight:600;margin-bottom:16px">Monthly GGR Projection</h2>
        <div style="display:flex;gap:4px;align-items:flex-end;height:120px;padding:24px 0 0">{bars}</div></div>'''

    # ── Market Breakdown ──
    mkt_rows = ""
    for mk in rev.get("market_breakdown", []):
        cap = mk.get("captured_players", 0)
        annual = mk.get("ggr_365d", 0)
        pct = mk.get("pct_of_total", 0)
        bar_w = max(2, int(pct))
        mkt_rows += f'''<div style="display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid var(--border)">
            <div style="width:60px;font-size:12px;font-weight:600;color:var(--text-bright)">{mk.get("market","").upper()}</div>
            <div style="flex:1;height:6px;background:var(--bg-input);border-radius:3px;overflow:hidden"><div style="width:{bar_w}%;height:100%;background:rgba(255,255,255,0.2);border-radius:3px"></div></div>
            <div style="width:90px;text-align:right;font-family:var(--mono);font-size:12px;color:var(--text-bright)">${annual:,.0f}</div>
            <div style="width:50px;text-align:right;font-size:11px;color:var(--text-muted)">{pct}%</div>
            <div style="width:80px;text-align:right;font-size:11px;color:var(--text-dim)">{cap:,} players</div></div>'''
    markets_card = f'<div class="card"><h2 style="font-size:15px;font-weight:600;margin-bottom:12px">Market Breakdown</h2>{mkt_rows}</div>'

    # ── Sensitivity Analysis ──
    sens_rows = ""
    for s in rev.get("sensitivity", []):
        is_current = s.get("delta_pct", 0) == 0
        bg = "background:rgba(255,255,255,0.03)" if is_current else ""
        fw = "font-weight:700" if is_current else ""
        dc = "var(--success)" if s.get("delta_pct", 0) > 0 else ("var(--danger)" if s.get("delta_pct", 0) < 0 else "var(--text-muted)")
        marker = " ← current" if is_current else ""
        sens_rows += f'<tr style="{bg}"><td style="padding:6px 12px;font-family:var(--mono);font-size:12px;{fw}">{s.get("rtp",0)}%{marker}</td><td style="padding:6px 12px;font-family:var(--mono);font-size:12px">{s.get("hold_pct",0)}%</td><td style="padding:6px 12px;font-family:var(--mono);font-size:12px">${s.get("ggr_365d",0):,.0f}</td><td style="padding:6px 12px;font-size:12px;color:{dc}">{s.get("delta_pct",0):+.1f}%</td></tr>'
    sensitivity_card = f'''<div class="card"><h2 style="font-size:15px;font-weight:600;margin-bottom:12px">Sensitivity Analysis — What if RTP changes?</h2>
        <table style="width:100%;border-collapse:collapse"><tr><th style="font-size:11px;color:var(--text-muted);padding:6px 12px;text-align:left">RTP</th><th style="font-size:11px;color:var(--text-muted);padding:6px 12px;text-align:left">Hold %</th><th style="font-size:11px;color:var(--text-muted);padding:6px 12px;text-align:left">Annual GGR</th><th style="font-size:11px;color:var(--text-muted);padding:6px 12px;text-align:left">Delta</th></tr>{sens_rows}</table></div>'''

    # ── Benchmark Comparison ──
    bench_rows = ""
    for b in rev.get("benchmarks", []):
        sim_bar = max(2, int(b.get("similarity_pct", 0)))
        bench_rows += f'''<div style="display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid var(--border)">
            <div style="width:120px;font-size:12px;font-weight:500;color:var(--text-bright)">{b.get("title","")}</div>
            <div style="width:60px;font-size:11px;color:var(--text-dim)">{b.get("volatility","")}</div>
            <div style="width:60px;font-size:11px;color:var(--text-muted)">{b.get("rtp",0)}%</div>
            <div style="flex:1;height:4px;background:var(--bg-input);border-radius:2px"><div style="width:{sim_bar}%;height:100%;background:rgba(255,255,255,0.2);border-radius:2px"></div></div>
            <div style="width:50px;text-align:right;font-size:11px;color:var(--text-muted)">{b.get("similarity_pct",0)}%</div>
            <div style="width:50px;text-align:right;font-size:11px;color:var(--text-dim)">{b.get("performance_vs_ours","")}</div></div>'''
    benchmark_card = f'<div class="card"><h2 style="font-size:15px;font-weight:600;margin-bottom:12px">Benchmark Comparison</h2>{bench_rows}</div>'

    # ── Investment Breakdown ──
    dev_cost = rev.get("total_dev_cost", 0)
    cert_cost = rev.get("cert_cost", 0)
    feature_cost = dev_cost - 45000 - 12000 - 5000 - cert_cost  # Reverse-calculate feature cost
    invest_card = f'''<div class="card"><h2 style="font-size:15px;font-weight:600;margin-bottom:12px">Investment Analysis</h2>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
            <div style="padding:8px;border:1px solid var(--border);border-radius:6px"><div style="font-size:11px;color:var(--text-muted)">Base Dev</div><div style="font-size:14px;font-weight:600;color:var(--text-bright)">$45,000</div></div>
            <div style="padding:8px;border:1px solid var(--border);border-radius:6px"><div style="font-size:11px;color:var(--text-muted)">Features</div><div style="font-size:14px;font-weight:600;color:var(--text-bright)">${max(0,feature_cost):,.0f}</div></div>
            <div style="padding:8px;border:1px solid var(--border);border-radius:6px"><div style="font-size:11px;color:var(--text-muted)">Art + Audio</div><div style="font-size:14px;font-weight:600;color:var(--text-bright)">$17,000</div></div>
            <div style="padding:8px;border:1px solid var(--border);border-radius:6px"><div style="font-size:11px;color:var(--text-muted)">Certification</div><div style="font-size:14px;font-weight:600;color:var(--text-bright)">${cert_cost:,.0f}</div></div>
        </div>
        <div style="margin-top:12px;padding:12px;background:rgba(255,255,255,0.03);border-radius:8px;display:flex;justify-content:space-between">
            <div><div style="font-size:11px;color:var(--text-muted)">Total Investment</div><div style="font-size:18px;font-weight:700;color:var(--text-bright)">${dev_cost:,.0f}</div></div>
            <div style="text-align:right"><div style="font-size:11px;color:var(--text-muted)">Net Profit (Year 1)</div><div style="font-size:18px;font-weight:700;color:{"var(--success)" if rev.get("ggr_365d",0)-dev_cost>0 else "var(--danger)"}">${rev.get("ggr_365d",0)-dev_cost:,.0f}</div></div>
        </div></div>'''

    # ── Operator Scenarios ──
    op_rows = ""
    for ops in rev.get("operator_scenarios", []):
        op_rows += f'''<div style="display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid var(--border)">
            <div style="width:80px;font-size:12px;font-weight:600;color:var(--text-bright)">{ops.get("type","").replace("_"," ").title()}</div>
            <div style="flex:1;font-family:var(--mono);font-size:13px;color:var(--text-bright)">${ops.get("ggr_365d",0):,.0f}</div>
            <div style="font-size:11px;color:var(--text-muted)">Margin: {ops.get("margin_pct",0)}%</div></div>'''
    ops_card = f'<div class="card"><h2 style="font-size:15px;font-weight:600;margin-bottom:12px">Operator Type Scenarios</h2>{op_rows}</div>'

    # ── Risk + Vol Profile ──
    cannibal = rev.get("cannibalization_risk", "?")
    cannibal_c = {"low":"var(--success)","medium":"var(--warning)","high":"var(--danger)"}.get(cannibal, "var(--text-muted)")
    risk_card = f'''<div class="card"><h2 style="font-size:15px;font-weight:600;margin-bottom:12px">Risk Profile</h2>
        <div style="margin-bottom:12px"><label style="font-size:11px">Cannibalization Risk</label><div style="font-size:16px;font-weight:600;color:{cannibal_c}">{cannibal.upper()}</div></div>
        <div style="margin-bottom:12px"><label style="font-size:11px">Theme Appeal</label><div style="font-size:16px;font-weight:600;color:var(--text-bright)">{rev.get("theme_appeal",1.0)}x</div></div>
        <div><label style="font-size:11px">Volatility Profile</label><p style="font-size:12px;color:var(--text-muted);margin-top:4px">{rev.get("volatility_profile","")}</p></div></div>'''

    return layout(f'''
    <div style="margin-bottom:20px"><a href="/job/{job_id}/files" style="color:var(--text-dim);font-size:12px;text-decoration:none">&larr; Back to {_esc(job["title"])}</a></div>
    <h2 class="page-title" style="margin-bottom:4px">&#128176; Revenue Dashboard</h2>
    <p style="color:var(--text-muted);font-size:12px;margin-bottom:24px">{_esc(job["title"])} — Financial Projections</p>
    {hero}{periods}{chart}
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
        <div>{markets_card}{invest_card}{ops_card}</div>
        <div>{sensitivity_card}{benchmark_card}{risk_card}</div>
    </div>
    <div style="margin:24px 0 40px"><a href="/job/{job_id}/files" class="btn btn-ghost">Back to files</a></div>''', "history")


# ─── ENGINE EXPORT (Phase 6B) ───

@app.route("/api/job/<job_id>/export")
@login_required
def api_export(job_id):
    """Generate and download engine export package (Unity/Godot/Generic)."""
    user = current_user(); db = get_db()
    job = db.execute("SELECT * FROM jobs WHERE id=? AND user_id=?", (job_id, user["id"])).fetchone()
    db.close()
    if not job or not job["output_dir"]:
        return "Not found", 404

    fmt = request.args.get("format", "unity").lower()
    if fmt not in ("unity", "godot", "generic"):
        return "Invalid format. Use: unity, godot, generic", 400

    od = Path(job["output_dir"])

    # Check if pre-generated ZIP exists
    slug = job["title"].lower().replace(" ", "_").replace("'", "")[:30]
    zip_name = f"{slug}_{fmt}_export.zip"
    cached = od / "09_export" / zip_name
    if cached.exists():
        return send_from_directory(cached.parent, cached.name, as_attachment=True,
                                    download_name=zip_name)

    # Generate on the fly
    try:
        from tools.export_engine import generate_export_package
        params = json.loads(job["params"]) if job["params"] else {}
        export_params = {
            "grid_cols": params.get("grid_cols", 5),
            "grid_rows": params.get("grid_rows", 3),
            "ways_or_lines": params.get("ways_or_lines", 243),
            "target_rtp": params.get("target_rtp", 96.0),
            "max_win": params.get("max_win_multiplier", 5000),
            "volatility": params.get("volatility", "medium"),
            "art_style": params.get("art_style", "Cinematic realism"),
            "markets": ", ".join(params.get("target_markets", [])) if isinstance(params.get("target_markets"), list) else params.get("target_markets", ""),
            "features": params.get("requested_features", []),
        }
        zip_path = generate_export_package(
            output_dir=str(od), format=fmt,
            game_title=job["title"], game_params=export_params,
        )
        zp = Path(zip_path)
        return send_from_directory(zp.parent, zp.name, as_attachment=True,
                                    download_name=zp.name)
    except Exception as e:
        return f"Export failed: {e}", 500


@app.route("/qdrant")
@login_required
def qdrant_status():
    try:
        from tools.qdrant_store import JurisdictionStore
        status = JurisdictionStore().get_status()
    except Exception as e:
        status = {"status":"ERROR","message":str(e),"jurisdictions":[],"total_vectors":0}
    bc = "badge-complete" if status["status"]=="ONLINE" else "badge-failed"
    jhtml = "".join(f'<div style="padding:8px 0;border-bottom:1px solid var(--border);font-size:13px">{j}</div>' for j in status.get("jurisdictions",[])) or '<div style="color:var(--text-muted);font-size:13px;padding:12px 0">No jurisdictions yet. Run a State Recon.</div>'
    return layout(f'''
    <h2 class="page-title" style="margin-bottom:24px">{ICON_DB} Qdrant Vector Database</h2>
    <div class="card"><h2>Connection <span class="badge {bc}" style="margin-left:8px">{status["status"]}</span></h2>
    <div class="row2" style="margin-top:12px"><div><label>Total Vectors</label><div style="font-size:20px;font-weight:600;color:var(--text-bright)">{status.get("total_vectors",0)}</div></div>
    <div><label>Jurisdictions</label><div style="font-size:20px;font-weight:600;color:var(--text-bright)">{len(status.get("jurisdictions",[]))}</div></div></div></div>
    <div class="card"><h2>Researched Jurisdictions</h2>{jhtml}</div>''', "qdrant")

# ─── REVIEWS (Web HITL) ───
@app.route("/reviews")
@login_required
def reviews_page():
    from tools.web_hitl import get_pending_reviews
    pending = get_pending_reviews()
    # Also get resolved reviews
    resolved = []
    try:
        db = get_db()
        resolved = db.execute(
            "SELECT r.*, j.title as job_title FROM reviews r JOIN jobs j ON r.job_id=j.id "
            "WHERE r.status!='pending' ORDER BY r.resolved_at DESC LIMIT 20"
        ).fetchall()
        db.close()
    except Exception:
        pass

    pending_html = ""
    for r in pending:
        pending_html += f'''<div class="history-item" style="grid-template-columns:1fr 140px 100px">
            <div><div class="history-title">{r["title"]}</div><div class="history-type">{r["job_title"]} &middot; {r["stage"]}</div></div>
            <div class="history-date">{r["created_at"][:16] if r["created_at"] else ""}</div>
            <div class="history-actions"><a href="/review/{r["id"]}" class="btn btn-primary btn-sm">Review</a></div>
        </div>'''
    if not pending_html:
        pending_html = '<div class="empty-state"><h3>No pending reviews</h3><p>Launch a pipeline in Interactive Mode to see checkpoints here.</p></div>'

    resolved_html = ""
    for r in resolved:
        r = dict(r)
        status = "Approved" if r.get("approved") else "Rejected"
        bc = "badge-complete" if r.get("approved") else "badge-failed"
        resolved_html += f'''<div class="history-item" style="grid-template-columns:1fr 100px 140px">
            <div><div class="history-title">{r["title"]}</div><div class="history-type">{r.get("job_title","")} &middot; {r.get("feedback","")[:50]}</div></div>
            <div><span class="badge {bc}">{status}</span></div>
            <div class="history-date">{r.get("resolved_at","")[:16]}</div>
        </div>'''

    return layout(f'''
    <h2 class="page-title" style="margin-bottom:24px">{ICON_REVIEW} Pipeline Reviews</h2>
    <div class="card"><h2 style="color:var(--text-bright)">Pending Reviews <span class="badge badge-running" style="margin-left:8px">{len(pending)}</span></h2>{pending_html}</div>
    {"<div class='card'><h2>Resolved</h2>" + resolved_html + "</div>" if resolved_html else ""}''', "reviews")


@app.route("/review/<review_id>")
@login_required
def review_detail(review_id):
    from tools.web_hitl import get_review
    import json as _json
    review = get_review(review_id)
    if not review:
        return "Review not found", 404

    files = _json.loads(_rget(review, "files","[]")) if _rget(review, "files") else []
    output_dir = _rget(review, "output_dir","")

    # Build file list with download links
    files_html = ""
    if files and output_dir:
        for f in files:
            fpath = Path(output_dir) / f
            if fpath.exists():
                ext = fpath.suffix.lower()
                # Show image previews inline
                if ext in (".png",".jpg",".jpeg",".webp"):
                    files_html += f'<div style="margin:8px 0"><div style="font-size:11px;color:var(--text-muted);margin-bottom:4px;font-family:Geist Mono,monospace">{f}</div><img src="/review/{review_id}/file/{f}" style="max-width:100%;border-radius:8px;border:1px solid var(--border)"></div>'
                else:
                    files_html += f'<div class="file-row"><a href="/review/{review_id}/file/{f}">{f}</a><span class="file-size">{fpath.stat().st_size/1024:.1f} KB</span></div>'

    if not files_html:
        files_html = '<div style="color:var(--text-muted);font-size:13px;padding:12px 0">No files to preview.</div>'

    already_resolved = review["status"] != "pending"
    form_html = ""
    if already_resolved:
        result = "Approved" if _rget(review, "approved") else "Rejected"
        form_html = f'<div class="card" style="border-color:var(--success) !important"><h2>Already {result}</h2><p style="color:var(--text-muted)">{_rget(review, "feedback","")}</p></div>'
    else:
        form_html = f'''<div class="card">
        <h2>Your Decision</h2>
        <form action="/api/review/{review_id}" method="POST">
            <label>Feedback / Art Changes / Notes</label>
            <textarea name="feedback" placeholder="e.g. Make the symbols darker, increase contrast on the wild symbol, add more gold accents..." rows="4"></textarea>
            <div style="display:flex;gap:12px;margin-top:8px">
                <button type="submit" name="action" value="approve" class="btn btn-primary" style="flex:1;padding:14px">Approve &amp; Continue</button>
                <button type="submit" name="action" value="reject" class="btn btn-ghost" style="flex:1;padding:14px;border-color:var(--danger);color:var(--danger)">Reject &amp; Revise</button>
            </div>
        </form></div>'''

    return layout(f'''
    <div style="margin-bottom:20px"><a href="/reviews" style="color:var(--text-dim);font-size:12px;text-decoration:none">&larr; Back to Reviews</a></div>
    <h2 class="page-title">{review["title"]}</h2>
    <p style="color:var(--text-muted);font-size:12px;margin-bottom:24px">{_rget(review, "job_title","")} &middot; Stage: {review["stage"]}</p>

    <div class="card"><h2>Summary</h2><div style="font-size:13px;line-height:1.7;white-space:pre-wrap">{review["summary"]}</div></div>
    <div class="card" style="padding:0;overflow:hidden"><div style="padding:16px 16px 8px"><h2 style="margin-bottom:8px">Generated Files</h2></div>{files_html}</div>
    {form_html}''', "reviews")


@app.route("/review/<review_id>/file/<path:fp>")
@login_required
def review_file(review_id, fp):
    from tools.web_hitl import get_review
    review = get_review(review_id)
    if not review or not _rget(review, "output_dir"):
        return "Not found", 404
    return send_from_directory(Path(review["output_dir"]), fp)


@app.route("/api/review/<review_id>", methods=["POST"])
@login_required
def api_submit_review(review_id):
    from tools.web_hitl import submit_review
    action = request.form.get("action","approve")
    feedback = request.form.get("feedback","")
    approved = (action == "approve")
    submit_review(review_id, approved=approved, feedback=feedback)
    return redirect("/reviews")


# ─── SETTINGS ───
@app.route("/settings")
@login_required
def settings_page():
    keys = {
        "OPENAI_API_KEY": {"label": "OpenAI API Key", "icon": "🧠", "desc": "GPT-5 reasoning agents, DALL-E 3 images, Vision QA", "required": True},
        "SERPER_API_KEY": {"label": "Serper API Key", "icon": "🔍", "desc": "Web search, patent search, trend radar, competitor teardown", "required": True},
        "ELEVENLABS_API_KEY": {"label": "ElevenLabs API Key", "icon": "🔊", "desc": "AI sound effect generation (13 core game sounds)", "required": False},
        "QDRANT_URL": {"label": "Qdrant URL", "icon": "🗃️", "desc": "Vector DB for regulation storage + knowledge base", "required": False},
        "QDRANT_API_KEY": {"label": "Qdrant API Key", "icon": "🔑", "desc": "Auth for Qdrant Cloud", "required": False},
        "GOOGLE_CLIENT_ID": {"label": "Google OAuth Client ID", "icon": "🔐", "desc": "Google sign-in", "required": True},
        "GOOGLE_CLIENT_SECRET": {"label": "Google OAuth Secret", "icon": "🔐", "desc": "Google sign-in", "required": True},
    }

    rows = ""
    for env_key, info in keys.items():
        val = os.getenv(env_key, "")
        is_set = bool(val) and val not in ("your-openai-key", "your-serper-key", "your-elevenlabs-key", "your-qdrant-key", "your-qdrant-url", "your-google-client-id", "your-google-client-secret")
        masked = val[:8] + "..." + val[-4:] if is_set and len(val) > 12 else ("Set" if is_set else "Not configured")
        bc = "badge-complete" if is_set else ("badge-failed" if info["required"] else "badge-queued")
        status = "Connected" if is_set else ("Required" if info["required"] else "Optional")
        rows += f'''<div class="file-row" style="padding:14px 16px;gap:16px">
            <div style="display:flex;align-items:center;gap:12px;flex:1">
                <span style="font-size:20px">{info["icon"]}</span>
                <div><div style="font-weight:600;color:var(--text-bright);font-size:13px">{info["label"]}</div>
                <div style="font-size:11px;color:var(--text-muted)">{info["desc"]}</div></div>
            </div>
            <div style="font-family:'Geist Mono',monospace;font-size:11px;color:var(--text-muted);min-width:120px">{masked}</div>
            <span class="badge {bc}">{status}</span>
        </div>'''

    return layout(f'''
    <h2 class="page-title">{ICON_SETTINGS} Settings</h2>
    <p style="color:var(--text-muted);font-size:13px;margin-bottom:24px">API keys and integrations. Configure in <code style="font-family:'Geist Mono',monospace;background:var(--bg-input);padding:2px 6px;border-radius:4px">.env</code> file.</p>
    <div class="card" style="padding:0;overflow:hidden"><div style="padding:16px 16px 8px"><h2>🔗 API Integrations</h2></div>{rows}</div>
    <div class="card"><h2>📋 Quick Setup</h2>
    <pre style="background:var(--bg-input);padding:16px;border-radius:8px;font-family:'Geist Mono',monospace;font-size:11px;color:var(--text);overflow-x:auto;line-height:1.8">
# Copy .env.example to .env and fill in your keys:
cp .env.example .env

# Required:
OPENAI_API_KEY=sk-...          # OpenAI (GPT-5 + DALL-E 3)
SERPER_API_KEY=...              # serper.dev (free tier: 2500 searches)

# Optional (Tier 2):
ELEVENLABS_API_KEY=...          # elevenlabs.io ($5/mo starter for SFX)

# Optional (State Recon):
QDRANT_URL=...                  # Qdrant Cloud or self-hosted
QDRANT_API_KEY=...
</pre></div>

    <div class="card"><h2>🏗️ Pipeline Version</h2>
    <div class="row2">
        <div><label>Version</label><div style="font-size:16px;font-weight:600;color:var(--text-bright)">v5.0</div></div>
        <div><label>Model</label><div style="font-size:16px;font-weight:700;color:var(--text-bright)">GPT-5 + GPT-5-mini</div></div>
    </div>
    <div style="margin-top:12px;font-size:12px;color:var(--text-dim);line-height:1.7">
        6 reasoning agents · 8 PDF deliverables · HTML5 prototype · AI sound design · Patent scanner · Cert planner
    </div></div>''', "settings")


# ─── API ───
@app.route("/api/pipeline", methods=["POST"])
@login_required
def api_launch_pipeline():
    user = current_user(); job_id = str(uuid.uuid4())[:8]
    params = {"theme":request.form["theme"],"target_markets":[m.strip() for m in request.form.get("target_markets","Georgia, Texas").split(",")],"volatility":request.form.get("volatility","medium"),"target_rtp":float(request.form.get("target_rtp",96)),"grid_cols":int(request.form.get("grid_cols",5)),"grid_rows":int(request.form.get("grid_rows",3)),"ways_or_lines":request.form.get("ways_or_lines","243"),"max_win_multiplier":int(request.form.get("max_win_multiplier",5000)),"art_style":request.form.get("art_style","Cinematic realism"),"requested_features":request.form.getlist("features"),"competitor_references":[r.strip() for r in request.form.get("competitor_references","").split(",") if r.strip()],"special_requirements":request.form.get("special_requirements",""),"enable_recon":request.form.get("enable_recon")=="on"}
    db = get_db(); db.execute("INSERT INTO jobs (id,user_id,job_type,title,params,status) VALUES (?,?,?,?,?,?)", (job_id,user["id"],"slot_pipeline",params["theme"],json.dumps(params),"queued")); db.commit(); db.close()
    params["interactive"] = request.form.get("interactive") == "on"
    _spawn_worker(job_id, "pipeline", json.dumps(params))
    return redirect(f"/job/{job_id}/logs")

@app.route("/api/recon", methods=["POST"])
@login_required
def api_launch_recon():
    user = current_user(); sn = request.form["state"].strip(); job_id = str(uuid.uuid4())[:8]
    db = get_db(); db.execute("INSERT INTO jobs (id,user_id,job_type,title,params,status) VALUES (?,?,?,?,?,?)", (job_id,user["id"],"state_recon",f"Recon: {sn}",json.dumps({"state":sn}),"queued")); db.commit(); db.close()
    _spawn_worker(job_id, "recon", sn)
    return redirect(f"/job/{job_id}/logs")

@app.route("/api/status/<job_id>")
@login_required
def api_job_status(job_id):
    # DB is the source of truth (shared across gunicorn workers + subprocesses)
    db = get_db()
    job = db.execute("SELECT status,current_stage,error FROM jobs WHERE id=?", (job_id,)).fetchone()
    db.close()
    if not job:
        return jsonify({"error": "Not found"}), 404
    return jsonify(dict(job))


@app.route("/api/logs/<job_id>")
@login_required
def api_log_stream(job_id):
    """SSE endpoint — streams live log lines from the worker subprocess."""
    log_path = LOG_DIR / f"{job_id}.log"

    def generate():
        # Wait for log file to appear (worker may still be starting)
        waited = 0
        while not log_path.exists() and waited < 15:
            time.sleep(0.5)
            waited += 0.5
            yield f"data: Waiting for worker to start...\n\n"
        if not log_path.exists():
            yield f"data: [ERROR] Log file not found for job {job_id}\n\n"
            return

        with open(log_path, "r") as f:
            # Send existing content first
            for line in f:
                yield f"data: {line.rstrip()}\n\n"
            # Then tail for new lines
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line.rstrip()}\n\n"
                else:
                    # Check if job is done
                    db = get_db()
                    job = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
                    db.close()
                    if job and job["status"] in ("complete", "failed"):
                        # Read any remaining lines
                        for remaining in f:
                            yield f"data: {remaining.rstrip()}\n\n"
                        yield f"data: [JOB {job['status'].upper()}]\n\n"
                        return
                    time.sleep(1)

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # Disable nginx buffering
    })


@app.route("/job/<job_id>/logs")
@login_required
def job_logs_page(job_id):
    db = get_db(); job = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone(); db.close()
    if not job: return "Not found", 404
    status = job["status"]
    badge_class = {"running":"badge-running","complete":"badge-complete","failed":"badge-failed"}.get(status,"badge-queued")
    files_btn = f'<a href="/job/{job_id}/files" class="btn btn-primary btn-sm">View Files</a>' if status == "complete" else ""
    stage_text = job["current_stage"] or ""

    # ── HTML part (f-string safe — no JS braces) ──
    html = f'''
    <div style="margin-bottom:16px"><a href="/history" style="color:var(--text-dim);font-size:12px;text-decoration:none" onmouseover="this.style.color='var(--text-bright)'" onmouseout="this.style.color='var(--text-dim)'">&larr; Back</a></div>
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
        <div>
            <h2 style="font-size:18px;font-weight:600;color:var(--text-bright);margin-bottom:4px;letter-spacing:-0.02em">{_esc(job["title"])}</h2>
            <div style="font-size:12px;color:var(--text-muted);display:flex;align-items:center;gap:8px"><span id="jobStatus" class="badge {badge_class}">{status}</span> <span id="jobStage" class="{"stage-shimmer" if status == "running" else ""}" style="font-size:12px">{stage_text}</span></div>
        </div>
        <div style="display:flex;gap:6px" id="actionBtns">
            <button onclick="clearLog()" class="btn btn-ghost btn-sm">Clear</button>
            <button onclick="scrollToBottom()" class="btn btn-ghost btn-sm">↓</button>
            {files_btn}
        </div>
    </div>
    <div id="jobData" data-job-id="{job_id}" data-status="{status}" style="display:none"></div>
    <div class="log-terminal">
        <div class="log-header">
            <span style="font-family:'Geist Mono',monospace">arkainbrain — {job_id}</span>
        </div>
        <div class="log-body" id="logContainer" style="overflow-y:auto;height:calc(100vh - 260px)"></div>
    </div>'''

    # ── JS part (plain string — no f-string, no brace escaping needed) ──
    js = '''<script>
    (function() {
        var jobData = document.getElementById('jobData');
        var JOB_ID = jobData.dataset.jobId;
        var initialStatus = jobData.dataset.status;
        var logEl = document.getElementById('logContainer');
        var autoScroll = true;
        var statusDone = (initialStatus === 'complete' || initialStatus === 'failed');

        logEl.addEventListener('scroll', function() {
            autoScroll = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 50;
        });

        window.scrollToBottom = function() { logEl.scrollTop = logEl.scrollHeight; autoScroll = true; };
        window.clearLog = function() { logEl.innerHTML = ''; };

        function colorize(text) {
            if (text.indexOf('FAILED') !== -1 || text.indexOf('ERROR') !== -1 || text.indexOf('BLOCKER') !== -1)
                return '<span style="color:#ef4444">' + text + '</span>';
            if (text.indexOf('COMPLETE') !== -1 || text.indexOf('complete') !== -1)
                return '<span style="color:#22c55e">' + text + '</span>';
            if (text.indexOf('WARN') !== -1)
                return '<span style="color:#eab308">' + text + '</span>';
            if (text.indexOf('Stage') !== -1)
                return '<span style="color:#fff;font-weight:500">' + text + '</span>';
            if (text.charAt(0) === '[' && text.indexOf(']') !== -1)
                return '<span style="color:#555">' + text + '</span>';
            return text;
        }

        // Poll DB status every 3s
        var statusPoll = setInterval(function() {
            if (statusDone) { clearInterval(statusPoll); return; }
            fetch('/api/status/' + JOB_ID).then(function(r) { return r.json(); }).then(function(d) {
                var badge = document.getElementById('jobStatus');
                var stage = document.getElementById('jobStage');
                if (d.current_stage) {
                    stage.textContent = d.current_stage;
                    stage.className = 'stage-shimmer';
                }
                if (d.status !== badge.textContent) {
                    badge.textContent = d.status;
                    badge.className = 'badge badge-' + (d.status === 'complete' ? 'complete' : d.status === 'failed' ? 'failed' : d.status === 'running' ? 'running' : 'queued');
                    if (d.status === 'complete') {
                        statusDone = true;
                        stage.className = '';
                        stage.textContent = 'Done';
                        document.getElementById('actionBtns').innerHTML += '<a href="/job/' + JOB_ID + '/files" class="btn btn-primary btn-sm">View Files</a>';
                    }
                    if (d.status === 'failed') {
                        statusDone = true;
                        stage.className = '';
                    }
                }
            }).catch(function() {});
        }, 3000);

        var evtSource = new EventSource('/api/logs/' + JOB_ID);
        evtSource.onmessage = function(e) {
            var line = e.data;
            logEl.innerHTML += colorize(line) + '\\n';
            if (autoScroll) window.scrollToBottom();
            if (line.indexOf('[JOB COMPLETE]') !== -1) {
                document.getElementById('jobStatus').className = 'badge badge-complete';
                document.getElementById('jobStatus').textContent = 'complete';
                statusDone = true;
                evtSource.close();
            }
            if (line.indexOf('[JOB FAILED]') !== -1) {
                document.getElementById('jobStatus').className = 'badge badge-failed';
                document.getElementById('jobStatus').textContent = 'failed';
                statusDone = true;
                evtSource.close();
            }
        };
        evtSource.onerror = function() { evtSource.close(); };
    })();
    </script>'''

    return layout(html + js, "history")


# ─── BACKGROUND WORKERS (subprocess-based) ───

# Track running subprocesses for status polling
_running_procs = {}  # job_id → Popen

def _cleanup_finished():
    """Remove completed subprocesses from tracking dict."""
    for jid in list(_running_procs):
        proc = _running_procs[jid]
        if proc.poll() is not None:
            _running_procs.pop(jid, None)

def _spawn_worker(job_id, job_type, *args):
    """Spawn a worker subprocess. No import locks, no deadlocks."""
    _cleanup_finished()
    worker_path = Path(__file__).parent / "worker.py"
    cmd = ["python3", "-u", str(worker_path), job_type, job_id] + list(args)
    env = {
        **os.environ,
        "DB_PATH": DB_PATH,
        "LOG_DIR": str(LOG_DIR),
        # ── Kill CrewAI tracing prompt ──
        "CREWAI_TELEMETRY_OPT_OUT": "true",
        "OTEL_SDK_DISABLED": "true",
        "CREWAI_TRACING_ENABLED": "false",
        "DO_NOT_TRACK": "1",
        # ── OpenAI SDK retry config ──
        "OPENAI_MAX_RETRIES": "5",
        "OPENAI_TIMEOUT": "120",
    }
    proc = subprocess.Popen(
        cmd, env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(Path(__file__).parent),
        start_new_session=True,     # Detach from gunicorn — survives web server restarts
    )
    _running_procs[job_id] = proc

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"ARKAINBRAIN — http://localhost:{port}")
    app.run(debug=os.getenv("FLASK_DEBUG","false").lower()=="true", host="0.0.0.0", port=port)
