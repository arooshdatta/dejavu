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
:root {
  --dv-bg:#0b0e10; --dv-surface:#121719; --dv-raised:#191f20; --dv-border:#293233;
  --dv-text:#eaf0e8; --dv-muted:#9aa7a0; --dv-lime:#c8f36a; --dv-cyan:#65d9e8;
  --dv-red:#ff8177; --dv-amber:#f2bd69; --dv-radius:10px;
}
#MainMenu, footer {visibility:hidden;}
.stApp {background:radial-gradient(ellipse at 78% -15%, #18231c 0, transparent 37%),var(--dv-bg);}
.block-container {padding:2rem clamp(1rem,3vw,3rem) 3rem;max-width:1600px;}
[data-testid="stSidebar"] {background:#0e1213;border-right:1px solid var(--dv-border);}
[data-testid="stSidebar"] > div:first-child {padding-top:1.2rem;}
.hero {padding:0 0 1.25rem;border-bottom:1px solid var(--dv-border);margin-bottom:1.4rem;}
.hero h1 {margin:0;font-size:clamp(2rem,4vw,3.1rem);letter-spacing:-.065em;font-weight:760;line-height:1.05;}
.hero h1 span {color:var(--dv-lime);}
.hero p {margin:.55rem 0 0;color:var(--dv-muted);font-size:1rem;max-width:760px;}
.eyebrow {color:var(--dv-lime);font:650 .72rem/1.2 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.13em;text-transform:uppercase;margin-bottom:.6rem;}
.pill {display:inline-flex;align-items:center;gap:.5rem;padding:.4rem .7rem;border-radius:6px;font-weight:650;font-size:.82rem;border:1px solid;margin-right:.7rem;}
.pill.stopped {background:#9aa7a010;color:#bbc4bf;border-color:#54605b;}
.pill.still {background:#c8f36a14;color:var(--dv-lime);border-color:#779342;}
.pill.settling {background:#f2bd6914;color:var(--dv-amber);border-color:#8e7041;}
.pill.moving {background:#ff817714;color:var(--dv-red);border-color:#8e514d;}
.pill.analyzing {background:#65d9e814;color:var(--dv-cyan);border-color:#377e85;}
.pill.error {background:#ff817714;color:var(--dv-red);border-color:#8e514d;}
[data-testid="stMetric"] {background:linear-gradient(145deg,#151b1b,#101515);border:1px solid var(--dv-border);border-radius:var(--dv-radius);padding:.8rem 1rem;}
[data-testid="stMetricLabel"] {color:var(--dv-muted);font-size:.76rem;}
[data-testid="stMetricValue"] {font:650 1.55rem/1.1 ui-monospace,SFMono-Regular,Consolas,monospace;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:var(--dv-radius);border-color:var(--dv-border);}
[data-testid="stTabs"] [role="tablist"] {gap:.35rem;border-bottom:1px solid var(--dv-border);}
[data-testid="stTabs"] button[role="tab"] {border-radius:6px 6px 0 0;color:var(--dv-muted);padding:.65rem .9rem;}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {color:var(--dv-lime);border-bottom:2px solid var(--dv-lime);}
button[kind="primary"] {background:var(--dv-lime);color:#11160b;border:1px solid var(--dv-lime);font-weight:700;}
button[kind="primary"]:hover {background:#d7ff83;border-color:#d7ff83;color:#11160b;}
button:focus-visible,input:focus-visible,textarea:focus-visible,[role="combobox"]:focus-visible {outline:2px solid var(--dv-cyan)!important;outline-offset:2px!important;}
[data-testid="stTextInput"] input,[data-testid="stSelectbox"] [role="combobox"],[data-testid="stMultiSelect"] [role="combobox"] {border-radius:7px;}
.step {background:linear-gradient(145deg,#161d1b,#111616);border:1px solid var(--dv-border);border-radius:var(--dv-radius);padding:1rem 1.1rem;min-height:100px;}
.step b {color:var(--dv-lime);font:650 .77rem ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.05em;}
[data-testid="stCaptionContainer"] {color:var(--dv-muted);}
@media (max-width:700px) {
  .block-container {padding:1rem .85rem 2rem;}
  .hero h1 {font-size:2rem;}
  [data-testid="stTabs"] [role="tablist"] {overflow-x:auto;}
  [data-testid="stTabs"] button[role="tab"] {padding:.55rem .6rem;font-size:.82rem;}
}
@media (prefers-reduced-motion:reduce) {*,*::before,*::after {scroll-behavior:auto!important;animation-duration:.01ms!important;transition-duration:.01ms!important;}}
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
        if st.button("Reset all memory", width="stretch", type="secondary"):
            st.session_state["confirm_reset"] = True
        if st.session_state.get("confirm_reset"):
            st.warning("This permanently clears tracked objects, events, and snapshot records.")
            confirm_reset = st.checkbox("I understand this cannot be undone", key="confirm_reset_ack")
            reset_col, cancel_col = st.columns(2)
            if reset_col.button("Confirm reset", type="primary", disabled=not confirm_reset, width="stretch"):
                mon.stop()
                engine.reset()
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
if not mon.running and start_clicked:
    mon.start()
    st.rerun()

# ---------------- header + live status ----------------
st.markdown('<div class="hero"><div class="eyebrow">LOCAL SCENE INTELLIGENCE / SYSTEM 01</div>'
            '<h1>👁️ Déjà <span>Vu</span></h1>'
            '<p>A camera with a memory. Track what changed, find where it went, and get an honest “not sure” when the evidence is unclear.</p>'
            '</div>', unsafe_allow_html=True)


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
    search = st.text_input("Search objects", placeholder="Filter by name, label, or zone", key="object_search")
    flt = st.radio("Show", ["all", "present", "moved", "missing", "removed", "unknown"], horizontal=True)
    if flt != "all":
        objs = [o for o in objs if o["status"] == flt]
    if search.strip():
        needle = search.casefold().strip()
        objs = [o for o in objs if needle in o["name"].casefold() or needle in o["label"].casefold() or needle in o["zone"].casefold()]
    if not objs:
        if state["objects"]:
            st.info("No objects match these filters. Try another status or search term.")
        else:
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
