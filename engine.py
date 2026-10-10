"""Deja Vu engine v2: detection, ML appearance embeddings, identity matching, memory, Ask."""
import os, math, shutil, threading
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import store

DATA = store.DATA
SNAPS, CROPS, EVID = (os.path.join(DATA, x) for x in ("snapshots", "crops", "events"))
for _d in (SNAPS, CROPS, EVID):
    os.makedirs(_d, exist_ok=True)

# ---- tunables ----
MIN_BLUR, MIN_LIGHT, MAX_LIGHT = 25.0, 40, 225
IGNORE = {"person"}
MIN_AREA, MAX_AREA = 0.003, 0.5
ALLOWED = ["bottle", "cup", "bowl", "book", "cell phone", "scissors", "earbuds", "remote",
           "keyboard", "laptop", "backpack", "handbag", "umbrella", "clock", "vase",
           "toothbrush", "apple", "banana", "orange"]
ALIASES = {"skateboard": "cell phone"}      # known detector confusion on the demo phone; edit/delete freely
ZONES = [["top-left", "top-center", "top-right"],
         ["middle-left", "center", "middle-right"],
         ["bottom-left", "bottom-center", "bottom-right"]]
COLORS = {"present": (46, 204, 113), "moved": (255, 159, 28), "missing": (241, 196, 15),
          "removed": (231, 76, 60), "unknown": (149, 165, 166), "added": (52, 152, 219)}
DEFAULT_CFG = {"model": "yolov8s.pt", "conf_min": 0.45, "allowed": list(ALLOWED), "remove_after": 2,
               "match_max_cost": 0.6, "move_dist": 0.12, "use_embeddings": True,
               "embed_model": "facebook/dinov2-small"}

# ---------------- models (loaded lazily, shared) ----------------
_INFER_LOCK = threading.Lock()
_DET = {}
_OVERRIDE = {"detector": None, "embedder": None}      # used only by tests
EMB = {"obj": None, "error": None, "tried": False}

class Embedder:
    """DINOv2 appearance embeddings: a learned 'fingerprint' of each object crop."""
    def __init__(self, name):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval()
    def __call__(self, crops_bgr):
        imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
        inputs = self.proc(images=imgs, return_tensors="pt")
        with self.torch.no_grad():
            out = self.model(**inputs).last_hidden_state[:, 0]
        out = self.torch.nn.functional.normalize(out, dim=-1)
        return out.cpu().numpy()

def get_detector(name):
    if _OVERRIDE["detector"] is not None:
        return _OVERRIDE["detector"]
    if name not in _DET:
        from ultralytics import YOLO
        _DET[name] = YOLO(name)
    return _DET[name]

def get_embedder(name):
    if _OVERRIDE["embedder"] is not None:
        return _OVERRIDE["embedder"]
    if EMB["obj"] is None and not EMB["tried"]:
        EMB["tried"] = True
        try:
            EMB["obj"] = Embedder(name)
        except Exception as ex:                      # graceful fallback to colour matching
            EMB["error"] = f"{type(ex).__name__}: {ex}"[:300]
    return EMB["obj"]

def embedding_status():
    if _OVERRIDE["embedder"] is not None or EMB["obj"] is not None:
        return "on"
    return f"off ({EMB['error']})" if EMB["error"] else "not loaded yet"

# ---------------- storage wrappers ----------------
load = store.load_state

def reset():
    with store.LOCK:
        for d in (SNAPS, CROPS, EVID):
            for f in os.listdir(d):
                try: os.remove(os.path.join(d, f))
                except OSError: pass
        store.reset_all()

rename = store.rename
feedback = store.feedback

# ---------------- vision helpers ----------------
def quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur, light = float(cv2.Laplacian(gray, cv2.CV_64F).var()), float(gray.mean())
    reasons = []
    if blur < MIN_BLUR: reasons.append("image is blurry")
    if light < MIN_LIGHT: reasons.append("too dark")
    if light > MAX_LIGHT: reasons.append("too bright")
    return {"blur": round(blur, 1), "light": round(light, 1), "reasons": reasons}

def center(b): return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
def area(b): return max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
def zone_of(cx, cy): return ZONES[min(2, int(cy * 3))][min(2, int(cx * 3))]

def overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return (ix * iy) / area(a)

def detect(model, img, allowed=None, conf_min=0.45):
    allowed = set(ALLOWED if allowed is None else allowed)
    h, w = img.shape[:2]
    res = model(img, verbose=False, conf=conf_min)[0]
    out = []
    for b in res.boxes:
        conf = float(b.conf[0])
        if conf < conf_min:
            continue
        raw = res.names[int(b.cls[0])]
        label = ALIASES.get(raw, raw)
        if label not in IGNORE and label not in allowed:
            continue
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        box = [x1 / w, y1 / h, x2 / w, y2 / h]
        if label not in IGNORE and not (MIN_AREA <= area(box) <= MAX_AREA):
            continue
        out.append({"label": label, "conf": conf, "box": box})
    return out

def crop_view(img, box):
    h, w = img.shape[:2]
    x1, y1 = max(0, int(box[0] * w)), max(0, int(box[1] * h))
    x2, y2 = min(w, int(box[2] * w)), min(h, int(box[3] * h))
    return img[y1:y2, x1:x2]

def color_hist(crop):
    if crop.size == 0:
        return [0.0] * 48
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 6], [0, 180, 0, 256]).flatten()
    return (hist / (hist.sum() + 1e-9)).tolist()

def cost(o, d):
    """Lower = more likely the same physical object."""
    if o["label"] != d["label"]:
        return 9.0
    c1, c2 = center(o["box"]), center(d["box"])
    pos = min(1.0, math.hypot(c1[0] - c2[0], c1[1] - c2[1]) / 1.414)
    col = float(cv2.compareHist(np.float32(o["hist"]).reshape(-1, 1),
                                np.float32(d["hist"]).reshape(-1, 1), cv2.HISTCMP_BHATTACHARYYA))
    ar = min(1.0, abs(math.log(area(o["box"]) / area(d["box"]))))
    if o.get("emb") and d.get("emb"):                       # ML appearance similarity
        cos = float(np.dot(o["emb"], d["emb"]))
        e = min(1.0, max(0.0, 2.0 * (1.0 - cos)))
        return 0.40 * e + 0.15 * col + 0.30 * pos + 0.15 * ar
    return 0.55 * col + 0.30 * pos + 0.15 * ar

def annotate(img, items):
    out = img.copy()
    h, w = out.shape[:2]
    scale = max(0.5, w / 1200)
    for it in items:
        x1, y1, x2, y2 = it["box"]
        r, g, b = COLORS.get(it["status"], (255, 255, 255))
        col = (b, g, r)
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        cv2.rectangle(out, p1, p2, col, max(2, int(3 * scale)))
        label = f'{it["name"]} - {it["status"]}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, 2)
        top = max(0, p1[1] - th - 8)
        cv2.rectangle(out, (p1[0], top), (p1[0] + tw + 8, top + th + 8), col, -1)
        cv2.putText(out, label, (p1[0] + 4, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * scale, (255, 255, 255), 2, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

# ---------------- memory ----------------
def new_object(s, d, ts, img):
    oid = s["next_obj"]; s["next_obj"] += 1
    n = s["counters"].get(d["label"], 0) + 1
    s["counters"][d["label"]] = n
    path = os.path.join(CROPS, f"{oid}.jpg")
    c = crop_view(img, d["box"])
    if c.size: cv2.imwrite(path, c)
    o = {"id": oid, "name": f"{d['label']} #{n}", "label": d["label"], "box": d["box"],
         "hist": d["hist"], "emb": d.get("emb"), "zone": zone_of(*center(d["box"])), "status": "present",
         "missing": 0, "first_seen": ts, "last_seen": ts, "crop": path,
         "conf": round(d["conf"], 2), "note": ""}
    s["objects"][str(oid)] = o
    return o

def add_event(s, typ, o, text, conf, sid, ts, before, after, bucket):
    eid = s["next_event"]; s["next_event"] += 1
    b = a = None
    if before and os.path.exists(before):
        b = os.path.join(EVID, f"{eid}_before.jpg"); shutil.copy(before, b)
    if after and os.path.exists(after):
        a = os.path.join(EVID, f"{eid}_after.jpg"); shutil.copy(after, a)
    ev = {"id": eid, "ts": ts, "type": typ, "object_id": o["id"], "name": o["name"],
          "text": text, "confidence": round(conf, 2), "snapshot_id": sid,
          "before": b, "after": a, "verdict": None}
    s["events"].append(ev); bucket.append(ev)

def has_missing():
    return store.count_status("missing") > 0

def process(img, cfg=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    q = quality(img)
    with _INFER_LOCK:
        all_dets = detect(get_detector(cfg["model"]), img, cfg["allowed"], cfg["conf_min"])
    persons = [d for d in all_dets if d["label"] in IGNORE]
    dets = [d for d in all_dets if d["label"] not in IGNORE]

    safe = img.copy()                      # privacy: blur people before storing
    for p in persons:
        r = crop_view(safe, p["box"])
        if r.size: r[:] = cv2.GaussianBlur(r, (51, 51), 0)
    usable = not q["reasons"]
    for d in dets:
        d["hist"] = color_hist(crop_view(safe, d["box"]))
    emb_used = False
    if usable and dets and cfg["use_embeddings"]:
        embedder = get_embedder(cfg["embed_model"])
        ok = [(i, crop_view(safe, d["box"])) for i, d in enumerate(dets)]
        ok = [(i, c) for i, c in ok if c.size and min(c.shape[:2]) >= 8]
        if embedder is not None and ok:
            try:
                with _INFER_LOCK:
                    vecs = embedder([c for _, c in ok])
                for (i, _), v in zip(ok, vecs):
                    dets[i]["emb"] = [round(float(x), 4) for x in v]
                emb_used = True
            except Exception as ex:
                EMB["error"] = f"{type(ex).__name__}: {ex}"[:300]

    with store.LOCK:
        s = store.load_state()
        sid = s["next_snap"]; s["next_snap"] += 1
        snap_path = os.path.join(SNAPS, f"{sid}.jpg")
        cv2.imwrite(snap_path, safe)
        s["snapshots"].append({"id": sid, "ts": ts, "path": snap_path, "usable": usable,
                               "person_present": bool(persons), "reasons": q["reasons"],
                               "blur": q["blur"], "light": q["light"]})
        base = {"quality": q, "usable": usable, "person_present": bool(persons), "events": [],
                "embeddings": emb_used, "snapshot_id": sid, "ts": ts}

        if not usable:
            store.save_state(s)
            return {**base, "image": annotate(safe, []),
                    "message": "Snapshot rejected (" + ", ".join(q["reasons"]) + "). Memory unchanged."}

        items, new_events = [], []
        if s["mode"] == "idle":
            for d in dets:
                o = new_object(s, d, ts, safe)
                items.append({"box": o["box"], "name": o["name"], "status": "present"})
            s["mode"] = "monitoring"
            store.save_state(s)
            return {**base, "image": annotate(safe, items), "baseline": True,
                    "message": f"Baseline saved with {len(dets)} objects."}

        pool = list(s["objects"].values())
        matched_o, matched_d, pairs, C = set(), set(), [], None
        if pool and dets:
            C = np.array([[cost(o, d) for d in dets] for o in pool])
            rows, cols = linear_sum_assignment(C)
            pairs = [(r, c) for r, c in zip(rows, cols) if C[r, c] <= cfg["match_max_cost"]]

        for r, c in pairs:
            o, d = pool[r], dets[c]
            oc, nc = center(o["box"]), center(d["box"])
            dist = math.hypot(oc[0] - nc[0], oc[1] - nc[1])
            new_zone = zone_of(*nc)
            conf = max(0.0, 1.0 - float(C[r, c]))
            tmp = os.path.join(CROPS, f"{o['id']}_new.jpg")
            cc = crop_view(safe, d["box"])
            if cc.size: cv2.imwrite(tmp, cc)
            if o["status"] == "removed":
                add_event(s, "returned", o, f"{o['name']} is back (now in {new_zone})", conf, sid, ts, o["crop"], tmp, new_events)
                status = "present"
            elif dist > cfg["move_dist"]:
                add_event(s, "moved", o, f"{o['name']} moved from {o['zone']} to {new_zone}", conf, sid, ts, o["crop"], tmp, new_events)
                status = "moved"
            else:
                status = "present"
            if os.path.exists(tmp): os.replace(tmp, o["crop"])
            o.update(box=d["box"], hist=d["hist"], emb=d.get("emb") or o.get("emb"), zone=new_zone,
                     status=status, missing=0, last_seen=ts, conf=round(conf, 2), note="")
            matched_o.add(o["id"]); matched_d.add(c)
            items.append({"box": o["box"], "name": o["name"], "status": status})

        for o in pool:
            if o["id"] in matched_o:
                continue
            if o["status"] == "removed":
                items.append({"box": o["box"], "name": o["name"], "status": "removed"})
                continue
            if any(overlap(o["box"], p["box"]) > 0.1 for p in persons):
                o["status"], o["note"] = "unknown", "A person was in front of its last position."
            else:
                o["missing"] += 1
                if o["missing"] >= cfg["remove_after"]:
                    o["status"] = "removed"
                    add_event(s, "removed", o, f"{o['name']} removed from {o['zone']}",
                              min(0.95, 0.5 + 0.2 * o["missing"]), sid, ts, o["crop"], snap_path, new_events)
                else:
                    o["status"] = "missing"
            items.append({"box": o["box"], "name": o["name"], "status": o["status"]})

        for i, d in enumerate(dets):
            if i in matched_d: continue
            o = new_object(s, d, ts, safe)
            add_event(s, "added", o, f"{o['name']} appeared in {o['zone']}", d["conf"], sid, ts, None, o["crop"], new_events)
            items.append({"box": o["box"], "name": o["name"], "status": "added"})

        store.save_state(s)
        return {**base, "image": annotate(safe, items), "events": new_events,
                "message": f"{len(new_events)} change(s) recorded."}

def summary():
    s = store.load_state()
    counts = {}
    for o in s["objects"].values():
        counts[o["status"]] = counts.get(o["status"], 0) + 1
    today = datetime.now().strftime("%Y-%m-%d")
    verdicts = [e["verdict"] for e in s["events"] if e["verdict"] in ("correct", "wrong")]
    return {"mode": s["mode"], "objects": len(s["objects"]), "counts": counts,
            "events": len(s["events"]), "events_today": sum(1 for e in s["events"] if e["ts"].startswith(today)),
            "snapshots": s["snapshot_count"], "judged": len(verdicts),
            "judged_correct": sum(1 for v in verdicts if v == "correct")}

# ---------------- ask ----------------
def fmt_time(ts):
    try:
        t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts
    clock = t.strftime("%I:%M %p").lstrip("0")
    secs = (datetime.now() - t).total_seconds()
    if secs < 60: rel = "just now"
    elif secs < 3600: rel = f"{int(secs // 60)} min ago"
    elif secs < 86400: rel = f"{int(secs // 3600)} h ago"
    else: rel = t.strftime("%d %b")
    return f"{clock}, {rel}"

STATUS_OF_EVENT = {"removed": "removed", "moved": "moved", "added": "added", "returned": "present"}

def _imgs(pairs):
    return [(p, c) for p, c in pairs if p and os.path.exists(p)]

def _card(name, status, line, images):
    return {"name": name, "status": status, "line": line, "images": _imgs(images)}

def _ans(headline, cards, abstained=False, conf=0.0, reason=None, detail=""):
    return {"headline": headline, "detail": detail, "cards": cards, "abstained": abstained,
            "confidence": round(conf, 2), "reason": reason}

def ask(question):
    s = load(); q = question.lower()
    if s["mode"] == "idle" or not s["snapshots"]:
        return _ans("No memory yet", [], True, 0.0, "no baseline",
                    "Start monitoring and keep the scene still, or take a baseline photo first.")
    last = s["snapshots"][-1]
    latest_img = [(last["path"], "latest snapshot")]
    detail_extra = "" if last["usable"] else f"Note: the latest snapshot was rejected ({', '.join(last['reasons'])})."
    objs = list(s["objects"].values())
    exact = sorted([o for o in objs if o["name"].lower() in q], key=lambda o: -len(o["name"]))
    by_label = [o for o in objs if o["label"] in q]

    if not exact and not by_label:
        if any(w in q for w in ("missing", "gone", "removed", "disappear", "left")):
            bad = [o for o in objs if o["status"] in ("removed", "missing", "unknown")]
            if not bad:
                return _ans("Nothing is missing", [_card("Latest view", "present",
                            f"All {len(objs)} tracked objects are visible.", latest_img)], False, 0.8, None, detail_extra)
            cards = [_card(o["name"], o["status"], f"Last seen in {o['zone']} · {fmt_time(o['last_seen'])}",
                           [(o["crop"], "last seen")]) for o in bad]
            return _ans(f"{len(bad)} object{'s' if len(bad) > 1 else ''} not in view", cards, False, 0.75, None, detail_extra)
        recent = list(reversed(s["events"][-5:]))
        if not recent:
            return _ans("No changes yet", [_card("Latest view", "present", "Nothing has changed since the baseline.", latest_img)],
                        False, 0.8, None, detail_extra)
        cards = [_card(e["name"], STATUS_OF_EVENT.get(e["type"], "present"),
                       f"{e['text']} · {fmt_time(e['ts'])}", [(e["after"] or e["before"], "evidence")]) for e in recent]
        return _ans(f"Last {len(recent)} change{'s' if len(recent) > 1 else ''}", cards, False, 0.8, None, detail_extra)

    if not exact and len(by_label) > 1:
        cards = [_card(o["name"], o["status"], f"{o['zone']} · last seen {fmt_time(o['last_seen'])}", [(o["crop"], "")])
                 for o in by_label]
        return _ans("Which one do you mean?", cards, True, 0.0, "ambiguous",
                    "I track several of those. Ask again using the full name, e.g. \"" + by_label[0]["name"] + "\".")

    o = exact[0] if exact else by_label[0]
    evs = [e for e in s["events"] if e["object_id"] == o["id"]]
    name = o["name"][0].upper() + o["name"][1:]
    if o["status"] == "unknown":
        return _ans(f"Not sure about {o['name']}",
                    [_card(o["name"], "unknown", f"Last seen in {o['zone']} · {fmt_time(o['last_seen'])}", [(o["crop"], "last seen")])],
                    True, 0.3, "occluded", o["note"] or "Its last position was covered.")
    if o["status"] == "removed":
        rem = [x for x in evs if x["type"] == "removed"]
        e = rem[-1] if rem else None
        imgs = [(e["before"], "last seen"), (e["after"], "when it was gone")] if e else [(o["crop"], "last seen")]
        return _ans(f"{name} was removed",
                    [_card(o["name"], "removed", f"Last seen in {o['zone']} · {fmt_time(o['last_seen'])}"
                           + (f" · confirmed gone {fmt_time(e['ts'])}" if e else ""), imgs)],
                    False, e["confidence"] if e else 0.5, None, detail_extra)
    if o["status"] == "missing":
        return _ans(f"{name} was not visible just now",
                    [_card(o["name"], "missing", f"Last seen in {o['zone']} · {fmt_time(o['last_seen'])}", [(o["crop"], "last seen")])],
                    True, 0.4, "unconfirmed", "I need one more capture to confirm it is gone.")
    mv = [x for x in evs if x["type"] in ("moved", "returned")]
    line = f"In {o['zone']} · last seen {fmt_time(o['last_seen'])}"
    if mv:
        line += f" · last change: {mv[-1]['text']}"
    return _ans(f"{name} is in {o['zone']}", [_card(o["name"], o["status"], line, [(o["crop"], "current")])],
                False, o["conf"], None, detail_extra)
