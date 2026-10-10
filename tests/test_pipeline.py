"""Pipeline tests with a scripted fake detector (no camera, no ML models needed).

The tests run inside a throw-away temporary folder, so they never touch your real data/ directory.
"""
import os
import tempfile

os.chdir(tempfile.mkdtemp(prefix="dejavu_test_"))   # must happen BEFORE importing the app modules

import cv2
import numpy as np
import pytest

import engine
import store

W, H = 640, 480
rng = np.random.default_rng(0)
NOISE = rng.integers(90, 140, (H, W, 3)).astype(np.uint8)       # textured background (passes the blur check)
COLORS = {1: (200, 60, 60), 2: (60, 200, 60), 3: (60, 60, 200)}


class _T:
    def __init__(self, v): self.v = v
    def __getitem__(self, i): return self.v[i]

class _Box:
    def __init__(self, cls, conf, xyxy): self.cls, self.conf, self.xyxy = _T([cls]), _T([conf]), _T([xyxy])

class _Res:
    names = {0: "person", 1: "bottle", 2: "cup", 3: "book", 4: "hot dog", 5: "skateboard"}
    def __init__(self, boxes): self.boxes = boxes

class ScriptedDetector:
    """Returns whatever boxes the test last asked for, like a perfect detector."""
    def __init__(self): self.boxes = []
    def __call__(self, img, verbose=False, conf=0.25): return [_Res(self.boxes)]

class FakeEmbedder:
    def __call__(self, crops):
        v = np.array([c.reshape(-1, 3).mean(0) for c in crops], dtype=np.float32) + 1e-3
        return v / np.linalg.norm(v, axis=1, keepdims=True)


DET = ScriptedDetector()
engine._OVERRIDE["detector"] = DET
CFG = {"use_embeddings": False}
BASE = [(1, (50, 50, 110, 170)), (2, (300, 200, 360, 260)), (3, (500, 330, 600, 400))]


def scene(objs, person=None, dark=False):
    """Draw coloured blocks and tell the fake detector where they are."""
    img = NOISE.copy()
    DET.boxes = []
    for cls, (x1, y1, x2, y2) in objs:
        if cls in COLORS:
            img[y1:y2, x1:x2] = COLORS[cls]
        DET.boxes.append(_Box(cls, 0.9, [x1, y1, x2, y2]))
    if person:
        DET.boxes.append(_Box(0, 0.95, list(person)))
    return (img * 0.1).astype(np.uint8) if dark else img


@pytest.fixture(autouse=True)
def clean_state():
    engine.reset()
    yield


def status_of(label):
    return next(o["status"] for o in store.load_state()["objects"].values() if o["label"] == label)


def test_baseline_registers_objects():
    r = engine.process(scene(BASE), CFG)
    assert r["message"].startswith("Baseline saved with 3 objects")
    assert store.load_state()["mode"] == "monitoring"


def test_removal_needs_two_captures_then_is_confirmed():
    engine.process(scene(BASE), CFG)
    r1 = engine.process(scene(BASE[1:]), CFG)
    assert r1["events"] == [] and status_of("bottle") == "missing"      # not confirmed yet
    r2 = engine.process(scene(BASE[1:]), CFG)
    assert [e["type"] for e in r2["events"]] == ["removed"]
    assert status_of("bottle") == "removed"


def test_move_is_reported_with_zones():
    engine.process(scene(BASE), CFG)
    moved = [BASE[0], (2, (500, 50, 560, 110)), BASE[2]]
    r = engine.process(scene(moved), CFG)
    assert [e["type"] for e in r["events"]] == ["moved"]
    assert "moved from center to top-right" in r["events"][0]["text"]


def test_unlisted_labels_are_ignored_and_skateboard_is_not_aliased():
    noisy = BASE[:1] + [(4, (400, 40, 460, 90)), (5, (150, 300, 230, 340))]

    cfg = {**CFG, "allowed": ["bottle", "skateboard"]}
    engine.process(scene(noisy), cfg)

    labels = sorted(o["label"] for o in store.load_state()["objects"].values())

    assert labels == ["bottle", "skateboard"]


def test_person_in_front_gives_unknown_not_removed():
    engine.process(scene(BASE), CFG)
    r = engine.process(scene(BASE[1:], person=(0, 0, 160, 220)), CFG)           # person covers the bottle's spot
    assert status_of("bottle") == "unknown" and r["events"] == []
    assert engine.ask("where is the bottle")["abstained"] is True


def test_dark_snapshot_is_rejected_and_memory_is_unchanged():
    engine.process(scene(BASE), CFG)
    before = store.load_state()["objects"]
    r = engine.process(scene(BASE[1:], dark=True), CFG)
    assert r["usable"] is False and "too dark" in r["quality"]["reasons"]
    assert store.load_state()["objects"].keys() == before.keys()
    assert status_of("bottle") == "present"


def test_people_are_blurred_in_stored_snapshots():
    person = (120, 280, 280, 460)          # a region with no objects in it, only background texture
    engine.process(scene(BASE, person=person), CFG)
    saved = cv2.imread(store.load_state()["snapshots"][-1]["path"])
    x1, y1, x2, y2 = person
    assert saved[y1:y2, x1:x2].std() < NOISE[y1:y2, x1:x2].std() * 0.5          # clearly smoother = blurred


def test_ask_answers_with_evidence_images():
    engine.process(scene(BASE), CFG)
    engine.process(scene(BASE[1:]), CFG)
    engine.process(scene(BASE[1:]), CFG)
    a = engine.ask("Where did the bottle go?")
    assert a["headline"] == "Bottle #1 was removed" and not a["abstained"]
    assert len(a["cards"][0]["images"]) == 2
    assert engine.ask("What changed?")["cards"][0]["images"]


def test_embeddings_path_runs_when_an_embedder_is_available():
    engine._OVERRIDE["embedder"] = FakeEmbedder()
    try:
        r = engine.process(scene(BASE), {"use_embeddings": True})
        assert r["embeddings"] is True
        assert store.load_state()["objects"]["1"]["emb"] is not None
    finally:
        engine._OVERRIDE["embedder"] = None


def test_database_persists_rename_feedback_and_reset():
    engine.process(scene(BASE), CFG)
    engine.process(scene(BASE[1:]), CFG)
    engine.process(scene(BASE[1:]), CFG)
    engine.rename(2, "my cup")
    engine.feedback(1, "correct")
    s = store.load_state()
    assert s["objects"]["2"]["name"] == "my cup"
    assert s["events"][0]["verdict"] == "correct"
    assert {t: len(store.table_rows(t)) for t in ("objects", "events", "snapshots")} == {"objects": 3, "events": 1, "snapshots": 3}
    engine.reset()
    assert engine.summary()["objects"] == 0
