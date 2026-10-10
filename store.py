"""SQLite storage for Deja Vu (replaces the old JSON file)."""
import os, json, sqlite3, threading
from contextlib import contextmanager

DATA = "data"
DB = os.path.join(DATA, "dejavu.db")
LOCK = threading.RLock()          # one writer at a time (UI thread + camera thread)
os.makedirs(DATA, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS snapshots(
    id INTEGER PRIMARY KEY, ts TEXT, path TEXT, usable INTEGER, person_present INTEGER,
    reasons TEXT, blur REAL, light REAL);
CREATE TABLE IF NOT EXISTS objects(
    id INTEGER PRIMARY KEY, name TEXT, label TEXT, box TEXT, hist TEXT, emb TEXT, zone TEXT,
    status TEXT, missing INTEGER, first_seen TEXT, last_seen TEXT, crop TEXT, conf REAL, note TEXT);
CREATE TABLE IF NOT EXISTS events(
    id INTEGER PRIMARY KEY, ts TEXT, type TEXT, object_id INTEGER, name TEXT, text TEXT,
    confidence REAL, snapshot_id INTEGER, before TEXT, after TEXT, verdict TEXT);
CREATE INDEX IF NOT EXISTS idx_events_obj ON events(object_id);
"""

@contextmanager
def db():
    c = sqlite3.connect(DB, timeout=15)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()

def init():
    os.makedirs(DATA, exist_ok=True)
    with db() as c:
        try:
            c.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        c.executescript(SCHEMA)

init()

def _obj(r):
    return {"id": r["id"], "name": r["name"], "label": r["label"], "box": json.loads(r["box"]),
            "hist": json.loads(r["hist"]), "emb": json.loads(r["emb"]) if r["emb"] else None,
            "zone": r["zone"], "status": r["status"], "missing": r["missing"],
            "first_seen": r["first_seen"], "last_seen": r["last_seen"], "crop": r["crop"],
            "conf": r["conf"], "note": r["note"] or ""}

def load_state(snap_limit=50):
    with db() as c:
        meta = {r["key"]: r["value"] for r in c.execute("SELECT key, value FROM meta")}
        objects = {str(r["id"]): _obj(r) for r in c.execute("SELECT * FROM objects ORDER BY id")}
        events = [dict(r) for r in c.execute("SELECT * FROM events ORDER BY id")]
        snaps = [dict(r) for r in c.execute("SELECT * FROM snapshots ORDER BY id DESC LIMIT ?", (snap_limit,))][::-1]
        for sn in snaps:
            sn["usable"] = bool(sn["usable"]); sn["person_present"] = bool(sn["person_present"])
            sn["reasons"] = json.loads(sn["reasons"] or "[]")
        n = c.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
    return {"mode": meta.get("mode", "idle"), "next_obj": int(meta.get("next_obj", 1)),
            "next_event": int(meta.get("next_event", 1)), "next_snap": int(meta.get("next_snap", 1)),
            "counters": json.loads(meta.get("counters", "{}")),
            "objects": objects, "events": events, "snapshots": snaps, "snapshot_count": n}

def save_state(s):
    with LOCK, db() as c:
        for k in ("mode", "next_obj", "next_event", "next_snap"):
            c.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, str(s[k])))
        c.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", ("counters", json.dumps(s["counters"])))
        for o in s["objects"].values():
            c.execute("INSERT OR REPLACE INTO objects VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (o["id"], o["name"], o["label"], json.dumps(o["box"]), json.dumps(o["hist"]),
                       json.dumps(o["emb"]) if o.get("emb") else None, o["zone"], o["status"], o["missing"],
                       o["first_seen"], o["last_seen"], o["crop"], o["conf"], o.get("note", "")))
        max_e = c.execute("SELECT COALESCE(MAX(id),0) FROM events").fetchone()[0]
        for e in s["events"]:
            if e["id"] > max_e:
                c.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                          (e["id"], e["ts"], e["type"], e["object_id"], e["name"], e["text"], e["confidence"],
                           e["snapshot_id"], e["before"], e["after"], e["verdict"]))
        max_s = c.execute("SELECT COALESCE(MAX(id),0) FROM snapshots").fetchone()[0]
        for sn in s["snapshots"]:
            if sn["id"] > max_s:
                c.execute("INSERT INTO snapshots VALUES(?,?,?,?,?,?,?,?)",
                          (sn["id"], sn["ts"], sn["path"], int(sn["usable"]), int(sn["person_present"]),
                           json.dumps(sn["reasons"]), sn["blur"], sn["light"]))

def rename(oid, name):
    with LOCK, db() as c:
        c.execute("UPDATE objects SET name=? WHERE id=?", (name, int(oid)))

def feedback(eid, verdict):
    with LOCK, db() as c:
        c.execute("UPDATE events SET verdict=? WHERE id=?", (verdict, int(eid)))

def reset_all():
    with LOCK, db() as c:
        for t in ("meta", "snapshots", "objects", "events"):
            c.execute(f"DELETE FROM {t}")

def count_status(status):
    with db() as c:
        return c.execute("SELECT COUNT(*) FROM objects WHERE status=?", (status,)).fetchone()[0]

def table_rows(name):
    cols = {"objects": "id, name, label, status, zone, conf, first_seen, last_seen",
            "events": "id, ts, type, name, text, confidence, verdict, snapshot_id",
            "snapshots": "id, ts, usable, person_present, blur, light, path"}[name]
    with db() as c:
        return [dict(r) for r in c.execute(f"SELECT {cols} FROM {name} ORDER BY id DESC")]

def events_for(object_id):
    with db() as c:
        return [dict(r) for r in c.execute("SELECT * FROM events WHERE object_id=? ORDER BY id DESC", (int(object_id),))]
