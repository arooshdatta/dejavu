# Changelog

## v0.1-demo (hackathon build)
- Automatic capture: a background monitor watches the camera and analyzes a snapshot once the scene has been still for a set time.
- SQLite memory (objects, events, snapshots) with CSV export.
- YOLO detection with a desk-object whitelist, plus DINOv2 appearance embeddings for identity matching (falls back to colour matching if unavailable).
- Event ledger (added / removed / moved / returned) with before/after evidence images.
- "Ask" tab: answers come from the ledger, with images, confidence and "not sure" abstention.
- Privacy: people are detected and blurred before any image is stored.
- Streamlit dashboard: live view, objects, timeline with feedback buttons, ask, data tables.
