# Evaluation protocol

> **Status: no results recorded yet.** Fill in the tables below only from runs you actually performed. Do not estimate.

## Setup to record for every run

| Item | Value |
|---|---|
| Date / commit or tag | |
| Hardware (laptop, CPU/GPU) | |
| Camera and resolution | |
| Lighting (describe) | |
| Detector model / confidence | |
| Embeddings on or off | |
| Stillness seconds / motion threshold | |

## Metrics

- **Event precision:** of the events the system reported, how many were correct. Use the ✅ / ❌ buttons on the Timeline tab; the Data tab computes it from your verdicts.
- **Event recall:** of the real changes you made, how many were reported.
- **Latency:** seconds from finishing a change to the event appearing.
- **False triggers per hour:** snapshots triggered (or events raised) while nothing was changed. Count the rows added to the **Snapshots** table (Data tab) over a quiet period; every automatic snapshot is one timestamped row.
- **Identity errors:** a swapped or duplicated object identity.
- **Abstention quality:** how often "not sure" occurred when the view really was blocked, and how often a wrong answer was given anyway.

## Scenarios

Run each scenario several times and record every outcome (including failures).

| Scenario | Procedure | Runs | Detected | Correct detail (zone, name) | Notes |
|---|---|---|---|---|---|
| Removal | Remove one object, keep still | | | | |
| Move | Move one object to another zone | | | | |
| Add | Place a new object | | | | |
| Return | Put a removed object back | | | | |
| Similar objects | Two similar objects, move one | | | | |
| Occlusion | Hold a hand in front of an object | | | | |
| Covered lens | Cover the camera, then ask | | | | |
| Dark scene | Switch the light off | | | | |
| Hand passes by | Wave a hand through, change nothing | | | | |
| Quiet period (10 min) | Change room light, let shadows pass | | | | |

## Baseline comparison

For the quiet period and hand-passes scenarios, also record how many snapshots a plain "motion happened" alert would have raised versus how many real events Déjà Vu recorded. State the counts you observed.

## Reporting rules

- Report counts (for example "7 of 9") together with the conditions, never a bare percentage.
- Include failures and say what caused them.
- State clearly that results come from a small, hand-run test in one environment and do not generalise.

## Threats to validity

- Small number of runs, one room, one camera, one set of objects.
- The tester knows what changed, which can bias verdicts.
- Settings were tuned on the same scene they were measured on.
