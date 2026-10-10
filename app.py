import os, time
import numpy as np
import cv2
import pandas as pd
import streamlit as st
import engine, store
from monitor import Monitor

st.set_page_config(page_title="Déjà Vu", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px;}
.hero h1 {margin: 0; font-size: 2.1rem; letter-spacing: -0.5px;}
.hero p {margin: 2px 0 10px 0; color: #9db4cc; font-size: 1.0rem;}
.pill {display: inline-block; padding: 5px 14px; border-radius: 999px; font-weight: 600; font-size: 0.9rem;
       border: 1px solid; margin-right: 10px;}
.pill.stopped {background:#6A7D9322; color:#b6c3d2; border-color:#6A7D93;}
.pill.still {background:#0E9E7C22; color:#2FFFD0; border-color:#0E9E7C;}
.pill.settling {background:#FFB35C22; color:#FFC983; border-color:#FFB35C;}
.pill.moving {background:#FF7A5C22; color:#FF9C85; border-color:#FF7A5C;}
.pill.analyzing {background:#25C8F022; color:#7fe0fb; border-color:#25C8F0;}
.pill.error {background:#E74C3C22; color:#ff8b7e; border-color:#E74C3C;}
div[data-testid="stMetric"] {background:#11284A; border:1px solid #1d3d68; border-radius:12px; padding:10px 14px;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 12px;}
.step {background:#11284A; border:1px solid #1d3d68; border-radius:12px; padding:14px 16px; height:100%;}
.step b {color:#25C8F0;}
</style>
""", unsafe_allow_html=True)

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
        left, right = st.columns([3, 2])
        status = EVENT_STATUS.get(e["type"], "present")
        left.markdown(f"{BADGE[status]} **{e['type'].upper()}**  \n{e['text']}")
        left.caption(f"{engine.fmt_time(e['ts'])} · confidence {int(e['confidence'] * 100)}%")
        imgs = [(e.get("before"), "before"), (e.get("after"), "after")]
        imgs = [(p, c) for p, c in imgs if p and os.path.exists(p)]
        if imgs:
            for col, (p, c) in zip(right.columns(len(imgs)), imgs):
                col.image(p, caption=c, width=150)
        if with_feedback:
            b1, b2, b3, _ = left.columns([1, 1, 1, 2])
            for col, v, label in ((b1, "correct", "✅"), (b2, "wrong", "❌"), (b3, "unsure", "❓")):
                if col.button(label, key=f"fb_{v}_{e['id']}", help=f"Mark as {v}"):
                    engine.feedback(e["id"], v)
                    st.rerun()
            if e.get("verdict"):
                left.caption(f"Your verdict: {e['verdict']}")


def show_answer(a):
    text = f"**{a['headline']}**" + (f"  \n{a['detail']}" if a.get("detail") else "")
    (st.warning if a["abstained"] else st.success)(text)
    for c in a["cards"]:
        with st.container(border=True):
            left, right = st.columns([3, 2])
            left.markdown(f"{BADGE.get(c['status'], '')} **{c['name']}** · {c['status']}")
            left.caption(c["line"])
            if c["images"]:
                for col, (p, cap) in zip(right.columns(len(c["images"])), c["images"]):
                    col.image(p, caption=cap or None, width=170)
    st.caption(f"Confidence {int(a['confidence'] * 100)}%" + (f" · reason: {a['reason']}" if a["reason"] else ""))


# ---------------- sidebar ----------------
with st.sidebar:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", width=90)
    st.markdown("### Déjà Vu")
    st.caption("A camera that remembers what changed.")

    start_clicked = False
    if mon.running:
        if st.button("⏹  Stop monitoring", width="stretch"):
            mon.stop()
            st.rerun()
    else:
        start_clicked = st.button("▶  Start monitoring", type="primary", width="stretch")

    with st.expander("📷 Camera & auto-capture", expanded=True):
        source = st.text_input("Camera", "0", help="0 = built-in webcam, 1 = second camera, or a phone IP-camera URL.")
        settle = st.slider("Capture after stillness (seconds)", 1.0, 15.0, 3.0, 0.5,
                           help="The scene must stay still this long before a snapshot is analyzed.")
        thresh = st.slider("Motion threshold", 1.0, 12.0, 3.0, 0.5,
                           help="Higher = needs bigger movement to count as motion. Watch the motion bar on the Live tab.")
    with st.expander("🎯 Detection"):
        model_name = st.selectbox("Model", ["yolov8s.pt", "yolov8m.pt", "yolov8n.pt"],
                                  help="Bigger = more accurate but slower.")
        conf_min = st.slider("Detection confidence", 0.25, 0.80, 0.45, 0.05)
        allowed = st.multiselect("Objects to track", engine.ALLOWED, default=engine.ALLOWED)
    with st.expander("🧬 Identity matching"):
        use_emb = st.toggle("Use AI appearance embeddings", value=True,
                            help="DINOv2 fingerprints help tell similar objects apart. Falls back to colour if unavailable.")
        strict = st.slider("Match tolerance", 0.40, 0.90, 0.60, 0.05,
                           help="Higher = more tolerant when deciding two sightings are the same object.")
        remove_after = st.slider("Captures to confirm 'removed'", 1, 4, 2)
        st.caption(f"Embeddings: {engine.embedding_status()}")
    with st.expander("🗄️ Data"):
        if st.button("Reset all memory", width="stretch"):
            mon.stop()
            engine.reset()
            for k in ("last_answer", "seen_snap"):
                st.session_state.pop(k, None)
            st.rerun()
    st.caption("🔒 People are detected and blurred before any image is stored.")

engine_cfg = {"model": model_name, "conf_min": conf_min, "allowed": allowed, "remove_after": remove_after,
              "match_max_cost": strict, "use_embeddings": use_emb}
mon.engine_cfg = engine_cfg
mon.cfg.update(source=source, settle_secs=settle, motion_thresh=thresh)
if not mon.running and start_clicked:
    mon.start()
    st.rerun()

# ---------------- header + live status ----------------
st.markdown('<div class="hero"><h1>👁️ Déjà Vu</h1>'
            '<p>A camera with a memory. It notices what changed, answers “where did it go?”, and says '
            '“not sure” instead of guessing.</p></div>', unsafe_allow_html=True)


@st.fragment(run_every=1.5)
def status_bar():
    text, css = STATE_TEXT.get(mon.state, ("", "stopped"))
    ago = f" · last snapshot {int(time.time() - mon.last_snapshot_at)}s ago" if mon.last_snapshot_at else ""
    st.markdown(f'<span class="pill {css}">● {text}</span><span style="color:#9db4cc">{mon.snapshots_taken} auto snapshots{ago}</span>',
                unsafe_allow_html=True)
    if mon.error:
        st.error(mon.error)
    res = mon.last_result
    if res and st.session_state.get("seen_snap") != res.get("snapshot_id"):
        st.session_state["seen_snap"] = res.get("snapshot_id")
        for e in res.get("events", []):
            st.toast(f"{e['type'].upper()}: {e['text']}", icon="🔔")
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
    @st.fragment(run_every=1)
    def live_view():
        state = engine.load()
        if not mon.running and state["mode"] == "idle":
            st.markdown("#### Get started in three steps")
            a, b, c = st.columns(3)
            a.markdown('<div class="step"><b>1 · Start</b><br>Press <i>Start monitoring</i> in the sidebar.</div>', unsafe_allow_html=True)
            b.markdown('<div class="step"><b>2 · Hold still</b><br>Keep the scene still for a few seconds. Déjà Vu takes the baseline by itself.</div>', unsafe_allow_html=True)
            c.markdown('<div class="step"><b>3 · Change something</b><br>Move, remove or add an object. Wait a moment, then ask “What changed?”.</div>', unsafe_allow_html=True)
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Live camera**")
            if mon.running and mon.preview:
                st.image(mon.preview)
                st.progress(min(1.0, mon.motion / max(0.1, mon.cfg["motion_thresh"] * 2)),
                            text=f"Motion level {mon.motion:.1f} (threshold {mon.cfg['motion_thresh']:.1f})")
            else:
                st.info("Camera is off. Press **Start monitoring** in the sidebar, or use manual capture below.")
        with right:
            st.markdown("**Last analysis**")
            res = mon.last_result
            if res:
                st.image(res["image"])
                (st.success if res["usable"] else st.error)(res["message"])
                for e in res.get("events", []):
                    show_event(e)
                with st.expander("Activity log"):
                    for t, msg in list(mon.log)[:12]:
                        st.caption(f"{t} · {msg}")
            else:
                st.caption("Nothing analyzed yet.")

    live_view()

    with st.expander("Manual capture (fallback): upload a photo or use the browser camera"):
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
            with st.spinner("Analyzing (first run downloads the models)..."):
                r = engine.process(img, engine_cfg)
            st.session_state["manual_result"] = r
            st.session_state["camn"] = n + 1
            st.rerun()
        mr = st.session_state.get("manual_result")
        if mr:
            (st.success if mr["usable"] else st.error)(mr["message"])
            st.image(mr["image"])
            for e in mr["events"]:
                show_event(e)

# ---------------- OBJECTS ----------------
with tab_obj:
    state = engine.load()
    objs = list(state["objects"].values())
    flt = st.radio("Show", ["all", "present", "moved", "missing", "removed", "unknown"], horizontal=True)
    if flt != "all":
        objs = [o for o in objs if o["status"] == flt]
    if not objs:
        st.info("No objects to show yet. Start monitoring and let the scene settle to take the baseline.")
    cols = st.columns(4)
    for i, o in enumerate(objs):
        with cols[i % 4]:
            with st.container(border=True):
                if o.get("crop") and os.path.exists(o["crop"]):
                    st.image(o["crop"], width=170)
                new = st.text_input("Name", o["name"], key=f"name{o['id']}", label_visibility="collapsed")
                if new and new != o["name"]:
                    engine.rename(o["id"], new)
                    st.rerun()
                st.markdown(f"{BADGE.get(o['status'], '')} **{o['status']}** · {o['zone']}")
                st.caption(f"last seen {engine.fmt_time(o['last_seen'])}")
                with st.expander("History"):
                    hist = store.events_for(o["id"])
                    if not hist:
                        st.caption("No changes yet.")
                    for h in hist[:6]:
                        st.caption(f"{engine.fmt_time(h['ts'])} · {h['text']}")

# ---------------- TIMELINE ----------------
with tab_time:
    state = engine.load()
    types = st.multiselect("Event types", ["added", "removed", "moved", "returned"],
                           default=["added", "removed", "moved", "returned"])
    limit = st.slider("Show latest", 5, 50, 15)
    evs = [e for e in reversed(state["events"]) if e["type"] in types][:limit]
    if not evs:
        st.info("No changes recorded yet.")
    for e in evs:
        show_event(e, with_feedback=True)

# ---------------- ASK ----------------
with tab_ask:
    st.markdown("Ask in plain language. Answers come from the saved memory, with evidence.")
    c1, c2, c3 = st.columns(3)
    if c1.button("What changed?", width="stretch"):
        st.session_state["last_answer"] = engine.ask("What changed recently?")
    if c2.button("What is missing?", width="stretch"):
        st.session_state["last_answer"] = engine.ask("What is missing?")
    with st.form("askform", clear_on_submit=False):
        q = st.text_input("Your question", placeholder="Where did the bottle go?")
        if st.form_submit_button("Ask", type="primary") and q:
            st.session_state["last_answer"] = engine.ask(q)
    if st.session_state.get("last_answer"):
        show_answer(st.session_state["last_answer"])

# ---------------- DATA ----------------
with tab_data:
    sm = engine.summary()
    m1, m2, m3 = st.columns(3)
    m1.metric("Events you judged", sm["judged"])
    m2.metric("Judged correct", sm["judged_correct"])
    m3.metric("Precision on judged events",
              f"{int(100 * sm['judged_correct'] / sm['judged'])}%" if sm["judged"] else "n/a",
              help="Use the ✅ / ❌ buttons on the Timeline tab to build this honestly.")
    for name in ("objects", "events", "snapshots"):
        rows = store.table_rows(name)
        st.markdown(f"**{name.capitalize()}** · {len(rows)} rows")
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", height=220)
        st.download_button(f"Download {name}.csv", df.to_csv(index=False).encode("utf-8"),
                           f"{name}.csv", "text/csv", key=f"dl_{name}")
