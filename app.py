import os, time
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import streamlit as st
import engine, store
from monitor import Monitor

st.set_page_config(page_title="Déjà Vu", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")

theme_css = Path(__file__).parent.joinpath("assets", "theme.css").read_text(encoding="utf-8")
st.markdown(f"<style>{theme_css}</style>", unsafe_allow_html=True)

BADGE = {"present": "🟢", "moved": "🟠", "missing": "🟡", "removed": "🔴", "unknown": "⚪", "added": "🔵"}
EVENT_STATUS = {"removed": "removed", "moved": "moved", "added": "added", "returned": "present"}
STATE_TEXT = {
    "stopped": ("Monitoring is off", "stopped"),
    "starting": ("Starting camera…", "analyzing"),
    "loading": ("Loading AI models (first time takes a minute)…", "analyzing"),
    "moving": ("Motion detected", "moving"),
    "settling": ("Waiting for the scene to settle…", "settling"),
    "still": ("Watching: scene is still", "still"),
    "analyzing": ("Analyzing snapshot…", "analyzing"),
    "error": ("Problem with the camera or models", "error"),
}


@st.cache_resource
def get_monitor():
    return Monitor()


mon = get_monitor()


# ---------------- reusable UI pieces ----------------
def show_event(e, with_feedback=False):
    with st.container(border=True):
        left, right = st.columns([1.35, 1])
        status = EVENT_STATUS.get(e["type"], "present")
        left.caption(f"EVENT {e['id']:04d} · {engine.fmt_time(e['ts'])}")
        left.markdown(f"### {BADGE[status]} {e['type'].title()}")
        left.write(e["text"])
        left.caption(f"Confidence · {int(e['confidence'] * 100)}%")
        imgs = [(e.get("before"), "before"), (e.get("after"), "after")]
        imgs = [(p, c) for p, c in imgs if p and os.path.exists(p)]
        if imgs:
            for col, (p, c) in zip(right.columns(len(imgs)), imgs):
                col.image(p, caption=c, width="stretch")
        if with_feedback:
            left.markdown("**Was this event accurate?**")
            b1, b2, b3 = left.columns(3)
            for col, v, label in ((b1, "correct", "✓ Correct"), (b2, "wrong", "✕ Incorrect"), (b3, "unsure", "? Unsure")):
                if col.button(label, key=f"fb_{v}_{e['id']}", help=f"Mark this event as {v}", width="stretch"):
                    engine.feedback(e["id"], v)
                    st.rerun()
            if e.get("verdict"):
                left.caption(f"Your assessment · {e['verdict']}")


def show_answer(a):
    text = f"**{a['headline']}**" + (f"  \n\n{a['detail']}" if a.get("detail") else "")
    (st.warning if a["abstained"] else st.success)(text)
    for c in a["cards"]:
        with st.container(border=True):
            left, right = st.columns([1.2, 1])
            left.markdown(f"### {BADGE.get(c['status'], '')} {c['name']}")
            left.caption(f"STATUS · {c['status'].upper()}")
            left.write(c["line"])
            if c["images"]:
                for col, (p, cap) in zip(right.columns(len(c["images"])), c["images"]):
                    col.image(p, caption=cap or None, width="stretch")
    st.caption(f"EVIDENCE CONFIDENCE · {int(a['confidence'] * 100)}%" + (f" · {a['reason']}" if a["reason"] else ""))


def section_heading(index, title, description):
    st.markdown(
        f'<div class="section-heading"><div class="eyebrow">{index}</div>'
        f'<h2>{title}</h2><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


# ---------------- sidebar ----------------
with st.sidebar:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", width=90)
    st.markdown("### Déjà Vu")
    st.caption("A camera that remembers what changed.")
    if st.session_state.pop("reset_notice", None):
        st.success("Scene memory cleared.")

    with st.expander("📷 Camera & auto-capture", expanded=True):
        st.caption("Choose the source and the stillness threshold used by the monitor.")
        source = st.text_input("Camera", "0", help="0 = built-in webcam, 1 = second camera, or a phone IP-camera URL.")
        settle = st.slider("Capture after stillness (seconds)", 1.0, 15.0, 3.0, 0.5,
                           help="The scene must stay still this long before a snapshot is analyzed.")
        thresh = st.slider("Motion threshold", 1.0, 12.0, 3.0, 0.5,
                           help="Higher = needs bigger movement to count as motion. Watch the motion bar on the Live tab.")
    with st.expander("🎯 Detection"):
        st.caption("Tune the detector and the object classes kept in scene memory.")
        model_name = st.selectbox("Model", ["yolov8s.pt", "yolov8m.pt", "yolov8n.pt"],
                                  help="Bigger = more accurate but slower.")
        conf_min = st.slider("Detection confidence", 0.25, 0.80, 0.45, 0.05)
        allowed = st.multiselect("Objects to track", engine.ALLOWED, default=engine.ALLOWED)
    with st.expander("🧬 Identity matching"):
        st.caption("These settings determine how sightings are matched to known objects.")
        use_emb = st.toggle("Use AI appearance embeddings", value=True,
                            help="DINOv2 fingerprints help tell similar objects apart. Falls back to colour if unavailable.")
        strict = st.slider("Match tolerance", 0.40, 0.90, 0.60, 0.05,
                           help="Higher = more tolerant when deciding two sightings are the same object.")
        remove_after = st.slider("Captures to confirm 'removed'", 1, 4, 2)
        st.caption(f"Embeddings: {engine.embedding_status()}")
    with st.expander("🗄️ Data"):
        st.caption("Remove the local object, event, and snapshot memory.")
        if st.button("Reset all memory", width="stretch", type="secondary"):
            st.session_state["confirm_reset"] = True
        if st.session_state.get("confirm_reset"):
            st.warning("This permanently clears tracked objects, events, and snapshot records.")
            confirm_reset = st.checkbox("I understand this cannot be undone", key="confirm_reset_ack")
            reset_col, cancel_col = st.columns(2)
            if reset_col.button("Confirm reset", type="primary", disabled=not confirm_reset, width="stretch"):
                mon.stop()
                engine.reset()
                st.session_state["reset_notice"] = True
                for k in ("last_answer", "seen_snap", "manual_result", "confirm_reset", "confirm_reset_ack"):
                    st.session_state.pop(k, None)
                st.rerun()
            if cancel_col.button("Cancel", width="stretch"):
                st.session_state.pop("confirm_reset", None)
                st.session_state.pop("confirm_reset_ack", None)
                st.rerun()
    st.caption("🔒 People are detected and blurred before any image is stored.")

engine_cfg = {"model": model_name, "conf_min": conf_min, "allowed": allowed, "remove_after": remove_after,
              "match_max_cost": strict, "use_embeddings": use_emb}
mon.engine_cfg = engine_cfg
mon.cfg.update(source=source, settle_secs=settle, motion_thresh=thresh)

# ---------------- header + live status ----------------
head, action = st.columns([3, 1.5], vertical_alignment="center")
with head:
    st.markdown('<div class="hero"><div class="eyebrow">LOCAL SCENE INTELLIGENCE / SYSTEM 01</div>'
                '<h1>👁️ Déjà <span>Vu</span></h1>'
                '<p>A camera with a memory. Track what changed, find where it went, and get an honest “not sure” when the evidence is unclear.</p>'
                '</div>', unsafe_allow_html=True)
with action:
    if mon.running:
        if st.button("Stop monitoring", icon="⏹", type="secondary", width="stretch"):
            mon.stop()
            st.rerun()
    elif st.button("Start monitoring", icon="▶", type="primary", width="stretch", disabled=not source.strip()):
        mon.start()
        st.rerun()
    if not mon.running and not source.strip():
        st.caption("Set a camera source in the sidebar to start monitoring.")


@st.fragment(run_every=1.5)
def status_bar():
    text, css = STATE_TEXT.get(mon.state, ("", "stopped"))
    ago = f" · last snapshot {int(time.time() - mon.last_snapshot_at)}s ago" if mon.last_snapshot_at else ""
    st.markdown(f'<span class="pill {css}">● {text}</span><span class="status-meta">{mon.snapshots_taken} automatic snapshots{ago}</span>',
                unsafe_allow_html=True)
    if mon.error:
        st.error(mon.error)
    res = mon.last_result
    if res and st.session_state.get("seen_snap") != res.get("snapshot_id"):
        st.session_state["seen_snap"] = res.get("snapshot_id")
        for e in res.get("events", []):
            st.toast(f"{e['type'].upper()}: {e['text']}", icon="🔔")
    with st.expander("Workspace metrics", expanded=False):
        sm = engine.summary()
        c = sm["counts"]
        cols = st.columns(5)
        cols[0].metric("Objects tracked", sm["objects"])
        cols[1].metric("Present", c.get("present", 0) + c.get("moved", 0))
        cols[2].metric("Needs attention", c.get("missing", 0) + c.get("removed", 0) + c.get("unknown", 0))
        cols[3].metric("Changes today", sm["events_today"])
        cols[4].metric("Snapshots", sm["snapshots"])


status_bar()

tab_live, tab_obj, tab_time, tab_ask, tab_data = st.tabs(["📡 Live", "🧩 Objects", "🕒 Timeline", "💬 Ask", "🗄️ Data"])

# ---------------- LIVE ----------------
with tab_live:
    section_heading("01 / MONITOR", "Live scene", "Camera status, motion feedback, and the latest evidence from your space.")
    @st.fragment(run_every=1)
    def live_view():
        state = engine.load()
        if not mon.running and state["mode"] == "idle":
            st.markdown("#### Establish a baseline")
            st.caption("Start monitoring above, or use manual capture to analyze a photo without a live camera.")
            a, b, c = st.columns(3)
            a.markdown('<div class="step"><b>01 · Connect</b><br>Start the configured camera from the workspace header.</div>', unsafe_allow_html=True)
            b.markdown('<div class="step"><b>02 · Set baseline</b><br>Keep the scene still. The first settled frame becomes the baseline.</div>', unsafe_allow_html=True)
            c.markdown('<div class="step"><b>03 · Track change</b><br>Move or remove an object, then let the scene settle again.</div>', unsafe_allow_html=True)
        left, right = st.columns([3, 2])
        with left:
            st.markdown("#### Camera feed")
            if mon.preview and mon.running:
                st.image(mon.preview, width="stretch")
                st.progress(min(1.0, mon.motion / max(0.1, mon.cfg["motion_thresh"] * 2)),
                            text=f"Motion {mon.motion:.1f} / threshold {mon.cfg['motion_thresh']:.1f}")
            elif mon.running:
                st.info("Connecting to camera and preparing the vision models…")
            else:
                st.info("Monitoring is stopped. Configure the camera in the sidebar, then start it above.")
        with right:
            st.markdown("#### Latest analysis")
            res = mon.last_result
            if res:
                st.image(res["image"], width="stretch")
                (st.success if res["usable"] else st.error)(res["message"])
                for e in res.get("events", []):
                    show_event(e)
            else:
                st.caption("No analysis is available yet. The latest result will appear here.")
            with st.expander("Monitor activity", expanded=bool(mon.error)):
                if mon.log:
                    for t, msg in list(mon.log)[:12]:
                        st.caption(f"{t} · {msg}")
                else:
                    st.caption("Monitor messages will appear here when camera activity begins.")

    live_view()

    with st.expander("Manual capture · analyze a photo"):
        st.caption("Uploads work at any time. The browser camera is available when automatic monitoring is stopped.")
        mode = st.radio("Source", ["Upload a photo", "Browser camera"], horizontal=True,
                        help="The browser camera can't be used while automatic monitoring is running.")
        n = st.session_state.setdefault("camn", 0)
        src = None
        if mode == "Upload a photo":
            src = st.file_uploader("Photo", type=["jpg", "jpeg", "png"], key=f"up{n}")
        elif mon.running:
            st.warning("Stop monitoring first, because the webcam can only be used by one program at a time.")
        else:
            src = st.camera_input("Camera", key=f"cam{n}")
        if src is not None and st.button("Analyze snapshot", type="primary"):
            img = cv2.imdecode(np.frombuffer(src.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                st.error("This image could not be opened. Try a valid JPG or PNG file.")
            else:
                with st.spinner("Analyzing (first run downloads the models)..."):
                    r = engine.process(img, engine_cfg)
                st.session_state["manual_result"] = r
                st.session_state["camn"] = n + 1
                st.rerun()
        mr = st.session_state.get("manual_result")
        if mr:
            (st.success if mr["usable"] else st.error)(mr["message"])
            st.image(mr["image"], width="stretch")
            for e in mr["events"]:
                show_event(e)

# ---------------- OBJECTS ----------------
with tab_obj:
    section_heading("02 / OBJECT MEMORY", "Tracked objects", "Search the current scene memory, rename an object, or inspect its event history.")
    state = engine.load()
    objs = list(state["objects"].values())
    search_col, filter_col, count_col = st.columns([2.2, 1.2, 1])
    search = search_col.text_input("Find object", placeholder="Name, type, or zone", key="object_search")
    flt = filter_col.selectbox("Status", ["all", "present", "moved", "missing", "removed", "unknown"])
    if flt != "all":
        objs = [o for o in objs if o["status"] == flt]
    if search.strip():
        needle = search.casefold().strip()
        objs = [o for o in objs if needle in o["name"].casefold() or needle in o["label"].casefold() or needle in o["zone"].casefold()]
    count_col.metric("Matching objects", len(objs))
    if not objs:
        if state["objects"]:
            st.info("No objects match these filters. Try another status or search term.")
        else:
            st.info("No objects to show yet. Start monitoring and let the scene settle to take the baseline.")
    cols = st.columns(3)
    for i, o in enumerate(objs):
        with cols[i % len(cols)]:
            with st.container(border=True):
                if o.get("crop") and os.path.exists(o["crop"]):
                    st.image(o["crop"], width="stretch")
                st.caption(f"OBJECT {o['id']:04d} · {o['label'].upper()}")
                new = st.text_input("Object name", o["name"], key=f"name{o['id']}")
                if new and new != o["name"]:
                    engine.rename(o["id"], new)
                    st.rerun()
                st.markdown(f"{BADGE.get(o['status'], '')} **{o['status'].upper()}** · {o['zone']}")
                st.caption(f"Last seen · {engine.fmt_time(o['last_seen'])}")
                with st.expander("History"):
                    hist = store.events_for(o["id"])
                    if not hist:
                        st.caption("No changes yet.")
                    for h in hist[:6]:
                        st.caption(f"{engine.fmt_time(h['ts'])} · {h['text']}")

# ---------------- TIMELINE ----------------
with tab_time:
    section_heading("03 / EVENT LEDGER", "Change timeline", "Review detected changes with before-and-after evidence and help improve precision with feedback.")
    state = engine.load()
    filter_col, limit_col = st.columns([2, 1])
    types = filter_col.multiselect("Event types", ["added", "removed", "moved", "returned"],
                           default=["added", "removed", "moved", "returned"])
    limit = limit_col.slider("Latest events", 5, 50, 15)
    evs = [e for e in reversed(state["events"]) if e["type"] in types][:limit]
    st.caption(f"Showing {len(evs)} of {len(state['events'])} recorded events")
    if not evs:
        if state["events"]:
            st.info("No events match the selected types. Adjust the filter to see more of the ledger.")
        else:
            st.info("No changes have been recorded yet. Events appear here after the scene baseline is set.")
    for e in evs:
        show_event(e, with_feedback=True)

# ---------------- ASK ----------------
with tab_ask:
    section_heading("04 / MEMORY QUERY", "Ask your scene", "Answers are assembled from the saved event ledger and include photographic evidence when available.")
    c1, c2 = st.columns(2)
    if c1.button("What changed?", width="stretch"):
        st.session_state["last_answer"] = engine.ask("What changed recently?")
    if c2.button("What is missing?", width="stretch"):
        st.session_state["last_answer"] = engine.ask("What is missing?")
    with st.form("askform", clear_on_submit=False):
        q = st.text_input("Ask about an object or recent change", placeholder="Where did the bottle go?")
        ask_submitted = st.form_submit_button("Ask memory", type="primary")
    if ask_submitted and q.strip():
        st.session_state["last_answer"] = engine.ask(q)
    elif ask_submitted:
        st.warning("Enter a question to search the saved memory.")
    if st.session_state.get("last_answer"):
        show_answer(st.session_state["last_answer"])

# ---------------- DATA ----------------
with tab_data:
    section_heading("05 / MEMORY TABLES", "Data & export", "Inspect the records behind the interface and download the current tables as CSV.")
    sm = engine.summary()
    m1, m2, m3 = st.columns(3)
    m1.metric("Events you judged", sm["judged"])
    m2.metric("Judged correct", sm["judged_correct"])
    m3.metric("Precision on judged events",
              f"{int(100 * sm['judged_correct'] / sm['judged'])}%" if sm["judged"] else "n/a",
              help="Use the ✅ / ❌ buttons on the Timeline tab to build this honestly.")
    data_columns = {
        "objects": ["id", "name", "label", "status", "zone", "conf", "first_seen", "last_seen"],
        "events": ["id", "ts", "type", "name", "text", "confidence", "verdict", "snapshot_id"],
        "snapshots": ["id", "ts", "usable", "person_present", "blur", "light", "path"],
    }
    data_tabs = st.tabs(["Objects", "Events", "Snapshots"])
    for container, name in zip(data_tabs, ("objects", "events", "snapshots")):
        with container:
            rows = store.table_rows(name)
            st.markdown(f"#### {name.capitalize()} <span class=\"record-count\">{len(rows)} records</span>",
                        unsafe_allow_html=True)
            df = pd.DataFrame(rows, columns=data_columns[name])
            if rows:
                st.dataframe(df, width="stretch", height=min(420, 120 + len(rows) * 35), hide_index=True)
            else:
                st.info(f"No {name} have been recorded yet.")
            st.download_button(f"Download {name}.csv", df.to_csv(index=False).encode("utf-8"),
                               f"{name}.csv", "text/csv", key=f"dl_{name}")
