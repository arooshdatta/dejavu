"""Background camera monitor: watches for motion, and analyzes automatically once the scene is still."""
import sys, time, threading
from collections import deque
import cv2
import numpy as np
import engine


def open_source(src):
    src = str(src).strip()
    if src.isdigit():
        idx = int(src)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap
    return cv2.VideoCapture(src)          # phone IP-camera URL or a video file


class Monitor:
    def __init__(self):
        self.cfg = {"source": "0", "settle_secs": 3.0, "motion_thresh": 3.0, "min_gap": 2.0,
                    "confirm_delay": 2.5}
        self.engine_cfg = {}
        self.stop_evt = threading.Event()
        self.thread = None
        self.state = "stopped"          # stopped / starting / loading / moving / settling / still / analyzing / error
        self.error = None
        self.motion = 0.0
        self.preview = None             # JPEG bytes of the latest live frame
        self.last_result = None
        self.last_snapshot_at = None
        self.snapshots_taken = 0
        self.log = deque(maxlen=40)

    # ---- control ----
    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self):
        if self.running:
            return
        self.stop_evt.clear()
        self.error = None
        self.state = "starting"
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_evt.set()
        if self.thread:
            self.thread.join(timeout=5)
        self.state = "stopped"

    def _log(self, msg):
        self.log.appendleft((time.strftime("%H:%M:%S"), msg))

    # ---- worker loop ----
    def _analyze(self, frame, reason):
        self.state = "analyzing"
        try:
            res = engine.process(frame, dict(self.engine_cfg))
            self.last_result = res
            self.last_snapshot_at = time.time()
            self.snapshots_taken += 1
            self._log(f"{reason}: {res['message']}")
        except Exception as ex:
            self._log(f"analysis error: {type(ex).__name__}: {str(ex)[:120]}")
            self.error = f"{type(ex).__name__}: {str(ex)[:200]}"

    def _run(self):
        cap = open_source(self.cfg["source"])
        if not cap.isOpened():
            self.state = "error"
            self.error = (f"Could not open camera '{self.cfg['source']}'. Close other apps that use the camera, "
                          "try 1 instead of 0, or paste a phone IP-camera URL.")
            return
        self._log("camera opened")
        try:                                  # load models up front so the first snapshot isn't slow
            self.state = "loading"
            engine.get_detector(self.engine_cfg.get("model", engine.DEFAULT_CFG["model"]))
            if self.engine_cfg.get("use_embeddings", True):
                engine.get_embedder(self.engine_cfg.get("embed_model", engine.DEFAULT_CFG["embed_model"]))
            self._log("models ready")
        except Exception as ex:
            self.state = "error"
            self.error = f"Model load failed: {type(ex).__name__}: {str(ex)[:200]}"
            cap.release()
            return

        hist = deque(maxlen=4)
        last_motion = time.time()
        last_snap = 0.0
        dirty = True                          # first still moment after start => baseline / first snapshot
        confirm_due = None
        fails = 0
        while not self.stop_evt.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if fails > 50:
                    self.state = "error"
                    self.error = "Camera stopped delivering frames."
                    break
                time.sleep(0.1)
                continue
            fails = 0
            now = time.time()
            small = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
            if len(hist) == hist.maxlen:
                self.motion = float(np.mean(cv2.absdiff(small, hist[0])))
            hist.append(small)
            th, c = frame.shape[:2], self.cfg
            pv = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
            okj, buf = cv2.imencode(".jpg", pv, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if okj:
                self.preview = buf.tobytes()

            moving = self.motion > c["motion_thresh"]
            if moving:
                last_motion, dirty, confirm_due = now, True, None
                self.state = "moving"
            else:
                self.state = "settling" if dirty else "still"

            if dirty and not moving and now - last_motion >= c["settle_secs"] and now - last_snap >= c["min_gap"]:
                self._analyze(frame, "scene settled")
                last_snap, dirty = time.time(), False
                confirm_due = time.time() + c["confirm_delay"] if engine.has_missing() else None
            elif confirm_due and not moving and now >= confirm_due:
                self._analyze(frame, "confirmation shot")
                last_snap = time.time()
                confirm_due = time.time() + c["confirm_delay"] if engine.has_missing() else None
            time.sleep(0.08)
        cap.release()
        if self.state != "error":
            self.state = "stopped"
