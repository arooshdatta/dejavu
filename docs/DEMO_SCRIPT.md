# Demo script (about 3 minutes)

## Before you go on stage
- Reset all memory, then run once so the models are cached.
- Fixed camera, plain background, steady light, 4 to 5 props that detect reliably.
- Close other apps that use the webcam. Charge the laptop and disable sleep and notifications.
- Have a recorded backup video ready and a phone hotspot nearby.

## Script

| Time | Presenter says / does | What the audience sees |
|---|---|---|
| 0:00 | "Things move and nobody notices. Déjà Vu gives a camera a memory." Press **Start monitoring** | Status pill: loading, then "Watching" |
| 0:20 | "It takes the baseline by itself once the scene is still." | Baseline saved in the activity log, objects appear |
| 0:40 | Teammate removes an object and keeps still | A "removed" notification and the Timeline entry with before/after images |
| 1:10 | Ask: **"Where did the bottle go?"** | Answer card with headline, last seen time, images |
| 1:40 | "It also knows when it does not know." Put a hand in front of an object, then cover the lens | Status "unknown" and "snapshot rejected", memory unchanged |
| 2:10 | Open the **Data** tab | SQLite tables, CSV export, precision from judged events |
| 2:40 | "Everything is local, and people are blurred before anything is saved." State limits honestly | Closing slide: roadmap |

## If something fails
- Camera does not open: use the **Manual capture** panel with photos, or replay the recorded video by entering its file path in the **Camera** box.
- A detection is wrong: say so, and show the ❌ button. That feedback is part of the design.
- Do not apologise for known limits. Name them first.
