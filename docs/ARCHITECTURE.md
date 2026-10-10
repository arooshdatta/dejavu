# Architecture

## Components

| File | Responsibility |
|---|---|
| `monitor.py` | Background thread. Reads frames, measures motion, decides *when* to analyze, schedules confirmation shots |
| `engine.py` | Quality gate, detection, privacy blur, embeddings, identity matching, event state machine, Ask, summary stats |
| `store.py` | SQLite schema and read/write helpers. A global lock serialises writers (UI thread and camera thread) |
| `app.py` | Streamlit UI. Reads and writes through `engine` and `store`; live parts auto-refresh with fragments |

The monitor and the UI run in the same process. The UI never touches the camera; it only reads the shared state.

## Auto-capture logic

1. Each frame is shrunk to 160×120, converted to grayscale and blurred.
2. **Motion** is the mean absolute difference between the current frame and the one four frames earlier (about half a second).
3. Motion above the threshold means the scene is `moving`, and the "dirty" flag is set.
4. When the scene has stayed below the threshold for `settle_secs` (default 3 s), at least 2 s after the previous snapshot, one snapshot is analyzed.
5. If any object is `missing` after that analysis, a **confirmation shot** is taken every 2.5 s (while the scene is still) until the objects are confirmed `removed` or found again.
6. With an empty memory, the first still moment becomes the **baseline**.

## Snapshot processing (`engine.process`)

1. **Quality gate:** reject if blurry (variance of the Laplacian), too dark or too bright. Rejected snapshots are stored but never alter memory.
2. **Detection:** YOLO boxes are filtered by the object whitelist, label aliases, confidence and box size.
3. **Privacy:** detected people are blurred in the copy that is saved.
4. **Features:** per detection, an HSV colour histogram and (if enabled) a DINOv2 embedding of the crop.
5. **Matching:** Hungarian assignment between known objects and detections.
6. **State machine:** see below.
7. **Persist:** objects, events and the snapshot are written to SQLite in one locked section.

### Matching cost (lower means more likely the same object)

With embeddings:

```
cost = 0.40 · e + 0.15 · colour + 0.30 · position + 0.15 · size
e    = min(1, 2 · (1 − cosine_similarity))
```

Without embeddings: `0.55 · colour + 0.30 · position + 0.15 · size`.

Detections with a different class label get a cost of 9 and never match. A pair is accepted only if its cost is at most the **match tolerance** (default 0.60). The weights were chosen by hand and are not tuned on a dataset.

### Object state machine

| Situation | Result |
|---|---|
| Matched, moved less than 12% of the image | `present` |
| Matched, moved more than 12% of the image | `moved` + `moved` event |
| Matched, previously `removed` | `present` + `returned` event |
| Not matched, a person overlaps its last position by more than 10% | `unknown` (no event) |
| Not matched, first time | `missing` |
| Not matched, `remove_after` consecutive times (default 2) | `removed` + `removed` event |
| Detection with no matching object | new object + `added` event |

### Confidence

Heuristic values, not calibrated probabilities. For matches it is `1 − cost`. For removals it starts at 0.5 and rises by 0.2 per confirming snapshot, capped at 0.95.

## Ask

A rule-based parser: it looks for an object name or class in the question, then for intent words such as *missing* or *changed*. Answers are assembled from ledger rows into cards (headline, status, last seen, evidence images). If the object is `unknown`, `missing` or ambiguous, the answer is marked as not certain and says why.

## Database schema (`data/dejavu.db`)

| Table | Columns |
|---|---|
| `meta` | key, value (mode, id counters) |
| `snapshots` | id, ts, path, usable, person_present, reasons, blur, light |
| `objects` | id, name, label, box, hist, emb, zone, status, missing, first_seen, last_seen, crop, conf, note |
| `events` | id, ts, type, object_id, name, text, confidence, snapshot_id, before, after, verdict |

Images are stored on disk under `data/snapshots`, `data/crops` and `data/events`, and referenced by path.

## Design decisions

- **Ledger answers, not generated ones.** Facts only come from database rows, so the system cannot hallucinate an event.
- **Abstain over guess.** `unknown` exists so an occluded object is never reported as removed.
- **Graceful degradation.** If the embedding model cannot load, matching falls back to colour and the UI says so.
- **Stillness gate.** Analyzing only after motion settles avoids snapshots with hands in the frame and gives each change a built-in persistence check.
- **One process, one database.** Simple to run and demo. A hosted version would split the monitor from the web UI and move to Postgres.
