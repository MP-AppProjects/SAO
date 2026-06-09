"""
sao_admin.py -- warstwa uwierzytelniania, bazy danych SQLite i panelu
administracyjnego dla SAO (System Analiz Openfield).

Wydzielone z generator.py. Modul jest samowystarczalny: korzysta wylacznie
ze stdlib oraz st.session_state. Importowany w generator.py przez
`from sao_admin import *`.

UWAGA: plik MUSI pozostac czystym ASCII (polskie znaki zapisane jako
sekwencje ucieczki unicode), tak samo jak generator.py.
"""
import streamlit as st
import sqlite3
import hashlib
import secrets
import uuid
import os
import datetime
import ipaddress
import json
import urllib.request
import urllib.error


_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sao_admin.db")

_MODULE_KEYS = {
    "dashboard":    "\U0001f3e0 Dashboard",
    "import":       "\U0001f4e5 Import danych",
    "project":      "\U0001f4c1 Projekt i S\u0142ownik",
    "prep":         "\U0001f6e0\ufe0f Przygotowanie Danych",
    "analyses":     "\U0001f4c8 Analizy i Tabele",
    "regression":   "\U0001f4c9 Regresja",
    "anova":        "\U0001f4ca ANOVA",
    "normality":    "\U0001f4d0 Testy Normalno\u015bci",
    "factor":       "\U0001f52c Analiza Czynnikowa",
    "cluster":      "\U0001f3af Skupienia i Segmentacja",
    "conjoint":     "\U0001f4ca Conjoint",
    "maxdiff":      "\U0001f522 MaxDiff",
    "wordcloud":    "\u2601\ufe0f Chmura S\u0142\u00f3w",
    "waves":        "\U0001f30a Por\u00f3wnanie fal",
    "export_excel": "\U0001f4be Eksport do Excela",
    "export_pptx":  "\U0001f4ca Eksport do PowerPoint",
    "export_word":  "\U0001f4c4 Eksport do Worda",
    "admin":        "\U0001f512 Panel admina",
}

_LABEL_TO_KEY = {v: k for k, v in _MODULE_KEYS.items()}

_DEFAULT_SETTINGS = {
    "idle_timeout_minutes": "60",
    "min_pw_length":        "10",
    "max_fail_attempts":    "5",
    "lockout_minutes":      "30",
    "rate_limit_window":    "15",
}


def _init_db_schema(conn):
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            password_salt TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL,
            created_by INTEGER,
            is_active INTEGER NOT NULL DEFAULT 1,
            expires_at TEXT,
            must_change_password INTEGER NOT NULL DEFAULT 0,
            failed_login_count INTEGER NOT NULL DEFAULT 0,
            locked_until TEXT,
            last_login_at TEXT,
            email TEXT,
            full_name TEXT
        );
        CREATE TABLE IF NOT EXISTS module_permissions (
            user_id INTEGER NOT NULL,
            module_key TEXT NOT NULL,
            granted INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, module_key)
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_token TEXT UNIQUE,
            ip_address TEXT,
            user_agent TEXT,
            geo_country TEXT,
            geo_city TEXT,
            started_at TEXT NOT NULL,
            last_seen_at TEXT,
            ended_at TEXT,
            login_success INTEGER NOT NULL DEFAULT 0,
            logout_reason TEXT,
            attempted_username TEXT
        );
        CREATE TABLE IF NOT EXISTS activity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_id INTEGER,
            module TEXT,
            action TEXT,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            ip_address TEXT
        );
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor_user_id INTEGER,
            target_user_id INTEGER,
            event_type TEXT NOT NULL,
            details_json TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS geo_cache (
            ip_address TEXT PRIMARY KEY,
            country TEXT,
            city TEXT,
            fetched_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE INDEX IF NOT EXISTS ix_activity_user      ON activity_log(user_id, created_at);
        CREATE INDEX IF NOT EXISTS ix_activity_module    ON activity_log(module, created_at);
        CREATE INDEX IF NOT EXISTS ix_sessions_user      ON sessions(user_id, started_at);
        CREATE INDEX IF NOT EXISTS ix_sessions_active    ON sessions(ended_at);
        CREATE INDEX IF NOT EXISTS ix_audit_target       ON audit_log(target_user_id, created_at);
    """)
    for k, v in _DEFAULT_SETTINGS.items():
        cur.execute("INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)", (k, v))
    conn.commit()


def _hash_password(pw, salt=None):
    if salt is None:
        salt = secrets.token_bytes(16)
    elif isinstance(salt, str):
        salt = bytes.fromhex(salt)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt, 260000)
    return h.hex(), salt.hex()


def _verify_password(pw, stored_hash, stored_salt):
    if not stored_hash or not stored_salt:
        return False
    h, _ = _hash_password(pw, stored_salt)
    return secrets.compare_digest(h, stored_hash)


def _now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat()


def _ensure_default_admin(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        pw_hash, pw_salt = _hash_password("admin")
        cur.execute(
            """INSERT INTO users(username, password_hash, password_salt, role,
                                 created_at, is_active, must_change_password)
               VALUES (?, ?, ?, 'admin', ?, 1, 1)""",
            ("admin", pw_hash, pw_salt, _now_iso()),
        )
        admin_id = cur.lastrowid
        for mkey in _MODULE_KEYS.keys():
            cur.execute(
                "INSERT OR IGNORE INTO module_permissions(user_id, module_key, granted) VALUES (?, ?, 1)",
                (admin_id, mkey),
            )
        cur.execute(
            "INSERT INTO audit_log(actor_user_id, target_user_id, event_type, details_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (admin_id, admin_id, "bootstrap_admin", json.dumps({"note": "domyslne konto admin/admin"}), _now_iso()),
        )
        conn.commit()


@st.cache_resource
def get_db():
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _init_db_schema(conn)
    _ensure_default_admin(conn)
    return conn


def _get_setting(key, default=None):
    try:
        row = get_db().execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default
    except Exception:
        return default


def _set_setting(key, value):
    get_db().execute(
        "INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )


def _get_setting_int(key, default):
    try:
        return int(_get_setting(key, str(default)))
    except (TypeError, ValueError):
        return default


def _get_client_ip():
    try:
        hdrs = st.context.headers
        xff = (hdrs.get("X-Forwarded-For") or "").split(",")[0].strip()
        if xff:
            return xff
        xri = hdrs.get("X-Real-IP")
        if xri:
            return xri.strip()
    except Exception:
        pass
    return "127.0.0.1"


def _get_user_agent():
    try:
        return (st.context.headers.get("User-Agent") or "")[:500]
    except Exception:
        return ""


def _is_private_ip(ip):
    try:
        a = ipaddress.ip_address(ip)
        return a.is_private or a.is_loopback or a.is_link_local or a.is_reserved
    except ValueError:
        return True


def _geo_lookup(ip):
    """Zwraca (country, city). Uzywa cache w DB. Dla prywatnych IP zwraca ("LAN", "sie\u0107 lokalna")."""
    if not ip:
        return (None, None)
    if _is_private_ip(ip):
        return ("LAN", "sie\u0107 lokalna")
    conn = get_db()
    row = conn.execute("SELECT country, city FROM geo_cache WHERE ip_address=?", (ip,)).fetchone()
    if row:
        return (row["country"], row["city"])
    country, city = (None, None)
    try:
        req = urllib.request.Request(
            "https://ipapi.co/" + ip + "/json/",
            headers={"User-Agent": "SAO/1.0"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            country = payload.get("country_name") or payload.get("country")
            city    = payload.get("city")
    except Exception:
        country, city = (None, None)
    conn.execute(
        "INSERT OR REPLACE INTO geo_cache(ip_address, country, city, fetched_at) VALUES(?, ?, ?, ?)",
        (ip, country, city, _now_iso()),
    )
    return (country, city)


def _get_user_by_name(username):
    return get_db().execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()


def _get_user_by_id(user_id):
    return get_db().execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()


def _load_user_perms(user_id, role):
    if role == "admin":
        return {k: True for k in _MODULE_KEYS.keys()}
    rows = get_db().execute(
        "SELECT module_key, granted FROM module_permissions WHERE user_id=?", (user_id,)
    ).fetchall()
    perms = {k: False for k in _MODULE_KEYS.keys()}
    for r in rows:
        perms[r["module_key"]] = bool(r["granted"])
    perms["admin"] = False
    return perms


def _set_module_perm(user_id, module_key, granted, actor_id=None):
    get_db().execute(
        """INSERT INTO module_permissions(user_id, module_key, granted) VALUES(?, ?, ?)
           ON CONFLICT(user_id, module_key) DO UPDATE SET granted=excluded.granted""",
        (user_id, module_key, int(bool(granted))),
    )
    get_db().execute(
        "INSERT INTO audit_log(actor_user_id, target_user_id, event_type, details_json, created_at) VALUES(?, ?, ?, ?, ?)",
        (actor_id, user_id, "set_perm",
         json.dumps({"module": module_key, "granted": bool(granted)}), _now_iso()),
    )


def _validate_password_policy(pw, username=""):
    min_len = _get_setting_int("min_pw_length", 10)
    if len(pw) < min_len:
        return "Has\u0142o musi mie\u0107 co najmniej " + str(min_len) + " znak\u00f3w."
    if not any(c.isdigit() for c in pw):
        return "Has\u0142o musi zawiera\u0107 co najmniej jedn\u0105 cyfr\u0119."
    if not any(c.isalpha() for c in pw):
        return "Has\u0142o musi zawiera\u0107 co najmniej jedn\u0105 liter\u0119."
    if username and username.lower() in pw.lower():
        return "Has\u0142o nie mo\u017ce zawiera\u0107 nazwy u\u017cytkownika."
    return None


def _create_session(user_id, ip, user_agent, success=True, attempted_username=None, reason=None):
    token = uuid.uuid4().hex
    country, city = _geo_lookup(ip) if success else (None, None)
    now = _now_iso()
    cur = get_db().execute(
        """INSERT INTO sessions(user_id, session_token, ip_address, user_agent,
                                geo_country, geo_city, started_at, last_seen_at,
                                login_success, logout_reason, attempted_username)
           VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, token if success else None, ip, user_agent,
         country, city, now, now if success else None,
         1 if success else 0, reason, attempted_username),
    )
    return cur.lastrowid, token, country, city


def _validate_session(token):
    if not token:
        return None
    row = get_db().execute(
        "SELECT * FROM sessions WHERE session_token=? AND ended_at IS NULL",
        (token,),
    ).fetchone()
    return row


def _end_session(session_id, reason="logout"):
    get_db().execute(
        "UPDATE sessions SET ended_at=?, logout_reason=?, session_token=NULL WHERE id=? AND ended_at IS NULL",
        (_now_iso(), reason, session_id),
    )


def _touch_session(session_id):
    get_db().execute("UPDATE sessions SET last_seen_at=? WHERE id=?", (_now_iso(), session_id))


def _log_activity(module, action, metadata=None):
    try:
        uid = st.session_state.get("current_user_id")
        sid = st.session_state.get("session_db_id")
        ip  = st.session_state.get("current_user_ip") or _get_client_ip()
        meta_json = None
        if metadata is not None:
            try:
                meta_json = json.dumps(metadata, ensure_ascii=True, default=str)[:2000]
            except Exception:
                meta_json = None
        get_db().execute(
            """INSERT INTO activity_log(user_id, session_id, module, action, metadata_json, created_at, ip_address)
               VALUES(?, ?, ?, ?, ?, ?, ?)""",
            (uid, sid, module, action, meta_json, _now_iso(), ip),
        )
    except Exception:
        pass


def _tracked_button(label, module_key, action, **kwargs):
    """Owijka na st.button ktora loguje klikniecie do activity_log."""
    clicked = st.button(label, **kwargs)
    if clicked:
        _log_activity(module_key, action)
    return clicked


def _user_can_access(module_key):
    if not st.session_state.get("authenticated"):
        return False
    if st.session_state.get("current_user_role") == "admin":
        return module_key == "admin"
    # Modul importu danych jest zawsze dostepny dla zalogowanego nie-admina
    # (bez niego nie da sie wczytac danych); nie podlega uprawnieniom per-user.
    if module_key == "import":
        return True
    perms = st.session_state.get("current_user_perms") or {}
    return bool(perms.get(module_key, False))


def _require_module_access(module_key):
    if not _user_can_access(module_key):
        st.error("\U0001f512 Brak uprawnie\u0144 do tego modu\u0142u. Skontaktuj si\u0119 z administratorem.")
        st.stop()


def _is_locked(user_row):
    lu = user_row["locked_until"] if user_row else None
    if not lu:
        return False
    try:
        return datetime.datetime.fromisoformat(lu) > datetime.datetime.utcnow()
    except Exception:
        return False


def _bump_failed_login(user_id):
    max_fail = _get_setting_int("max_fail_attempts", 5)
    lock_min = _get_setting_int("lockout_minutes", 30)
    conn = get_db()
    row = conn.execute("SELECT failed_login_count FROM users WHERE id=?", (user_id,)).fetchone()
    new_count = (row["failed_login_count"] if row else 0) + 1
    locked_until = None
    if new_count >= max_fail:
        locked_until = (datetime.datetime.utcnow() + datetime.timedelta(minutes=lock_min)).replace(microsecond=0).isoformat()
    conn.execute(
        "UPDATE users SET failed_login_count=?, locked_until=? WHERE id=?",
        (new_count, locked_until, user_id),
    )
    return new_count, locked_until


def _reset_failed_login(user_id):
    get_db().execute(
        "UPDATE users SET failed_login_count=0, locked_until=NULL, last_login_at=? WHERE id=?",
        (_now_iso(), user_id),
    )


def _attempt_login(username, password, ip, user_agent):
    """Zwraca (status, payload). status: 'ok'|'bad_credentials'|'locked'|'inactive'|'expired'."""
    user = _get_user_by_name(username)
    if not user:
        _create_session(None, ip, user_agent, success=False,
                        attempted_username=username, reason="unknown_user")
        return ("bad_credentials", None)
    if _is_locked(user):
        return ("locked", user["locked_until"])
    if not user["is_active"]:
        return ("inactive", None)
    if user["expires_at"]:
        try:
            if datetime.datetime.fromisoformat(user["expires_at"]) < datetime.datetime.utcnow():
                return ("expired", user["expires_at"])
        except Exception:
            pass
    if not _verify_password(password, user["password_hash"], user["password_salt"]):
        _bump_failed_login(user["id"])
        _create_session(user["id"], ip, user_agent, success=False,
                        attempted_username=username, reason="bad_password")
        return ("bad_credentials", None)
    _reset_failed_login(user["id"])
    session_id, token, country, city = _create_session(user["id"], ip, user_agent, success=True)
    return ("ok", {"user": user, "session_id": session_id, "token": token,
                   "country": country, "city": city})

__all__ = [
    '_DB_PATH',
    '_MODULE_KEYS',
    '_LABEL_TO_KEY',
    '_DEFAULT_SETTINGS',
    '_init_db_schema',
    '_hash_password',
    '_verify_password',
    '_now_iso',
    '_ensure_default_admin',
    'get_db',
    '_get_setting',
    '_set_setting',
    '_get_setting_int',
    '_get_client_ip',
    '_get_user_agent',
    '_is_private_ip',
    '_geo_lookup',
    '_get_user_by_name',
    '_get_user_by_id',
    '_load_user_perms',
    '_set_module_perm',
    '_validate_password_policy',
    '_create_session',
    '_validate_session',
    '_end_session',
    '_touch_session',
    '_log_activity',
    '_tracked_button',
    '_user_can_access',
    '_require_module_access',
    '_is_locked',
    '_bump_failed_login',
    '_reset_failed_login',
    '_attempt_login',
]
