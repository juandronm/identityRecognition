"""
Real-time webcam face recognition using InsightFace.
  - RetinaFace  : face detection + 5-point landmarks
  - ArcFace R50 : 512-d face embedding (buffalo_l pack)
  - ONNX Runtime: CUDAExecutionProvider (falls back to CPU automatically)

Enrollment:
  Place photos in  faces_db/<YourName>/photo.jpg  before running.
  Press  r  during the session to enroll a new face live from the webcam.
  Press  q  to quit.
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path

# Resolve paths relative to this file so the script works regardless of the
# directory it is launched from.
_HERE = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# GPU bootstrap — must run before any onnxruntime import.
#
# onnxruntime-gpu's CUDA provider DLL needs cudart/cublas/cudnn DLLs to be
# findable via the Windows DLL search path.  The CUDA Toolkit installs them
# to the system PATH automatically.  Without the toolkit, we attempt to
# locate them inside PyTorch's bundled copy and pre-load them manually so
# the CUDAExecutionProvider can initialise.
# ---------------------------------------------------------------------------
def _bootstrap_cuda() -> bool:
    try:
        import ctypes
        import torch

        torch_lib = Path(torch.__file__).parent / "lib"
        if not torch_lib.exists():
            return False

        # Add to Windows DLL search path (process-wide, affects LoadLibrary)
        if platform.system() == "Windows" and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(torch_lib))
        os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")

        # Pre-load CUDA runtime DLLs so the onnxruntime CUDA provider finds them
        for dll in ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll"]:
            dll_path = torch_lib / dll
            if dll_path.exists():
                ctypes.WinDLL(str(dll_path))

        # Warm up the CUDA driver (creates a CUDA context in this process)
        if torch.cuda.is_available():
            torch.cuda.init()
            return True
    except Exception as e:
        print(f"[WARN] CUDA bootstrap failed: {e}")
    return False


_CUDA_AVAILABLE = _bootstrap_cuda()

import cv2
import numpy as np
from numpy.linalg import norm
from PIL import Image, ImageOps
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ---------------------------------------------------------------------------
# Tunables — adjust these without touching the rest of the script
# ---------------------------------------------------------------------------
FACES_DB      = _HERE.parent / "faces_db"   # sibling of src/ — safe regardless of cwd
CAMERA_INDEX  = 0            # change to 1, 2 … if your webcam is not the default device
DET_SIZE      = (1920, 1920) # GPU: upscales 720p 1.5×, stops 0.67× downscale on 1080p
                              # CPU: use (640,640) or (320,320) for acceptable FPS
MIN_DET_SCORE         = 0.60  # enrollment gate — strict
MIN_DET_SCORE_RUNTIME = 0.35  # runtime gate — distant/angled faces score lower
MIN_SHARPNESS         = 50.0  # enrollment gate — strict
MIN_SHARPNESS_RUNTIME = 10.0  # runtime gate — distant faces are naturally blurrier
SIM_THRESHOLD = 0.45
ENROLL_FRAMES = 5            # good-quality frames to capture during live enrollment

ZOOM_CROP     = 0.55  # center-crop fraction for 2nd detection pass (0 = disabled)
TRACK_MAX_AGE = 0.4   # seconds to hold a track after detection disappears (FPS-independent)
TRACK_MIN_IOU = 0.30  # min bounding-box overlap to match a detection to an existing track
INFER_EVERY   = 1     # run full inference every N frames; display uses tracker for skipped frames

# Display window — inference always runs on the full camera frame for accuracy;
# only the rendered output is scaled to this size before imshow.
# Set to None to use the camera's native resolution without scaling.
DISPLAY_SIZE  = (1280, 720)  # (width, height) of the OpenCV window

# Colors (BGR)
COLOR_KNOWN   = (0, 220, 0)
COLOR_UNKNOWN = (0, 0, 220)
COLOR_KP      = (255, 100, 0)
FONT          = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# InsightFace app
# ---------------------------------------------------------------------------
def build_app() -> FaceAnalysis:
    import onnxruntime as ort

    cuda_in_ort = "CUDAExecutionProvider" in ort.get_available_providers()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if cuda_in_ort
        else ["CPUExecutionProvider"]
    )
    ctx_id = 0 if cuda_in_ort else -1

    if cuda_in_ort:
        print("[INFO] Running on GPU (CUDA)")
    elif _CUDA_AVAILABLE:
        print("[WARN] GPU detected but onnxruntime CUDA provider could not initialise.")
        print("       Install the NVIDIA CUDA Toolkit to fix this:")
        print("       https://developer.nvidia.com/cuda-downloads")
        print("[INFO] Falling back to CPU.")
    else:
        print("[INFO] No GPU detected — running on CPU.")

    # CPU performance tuning:
    # - DET_SIZE=(320,320): 4× fewer pixels than 640 → roughly 3–4× faster detection.
    #   Minimum detectable face ~30px in a 1280px-wide frame (medium–close range).
    # - ZOOM_CROP=0: disables the second app.get() call that doubles per-frame cost.
    #   On GPU, ZOOM_CROP helps catch distant faces; on CPU it just halves FPS.
    # Install the NVIDIA CUDA Toolkit to unlock DET_SIZE=1920 + ZOOM_CROP for
    # full distance-detection performance.
    global DET_SIZE, ZOOM_CROP
    global INFER_EVERY
    if not cuda_in_ort:
        if DET_SIZE[0] > 320:
            DET_SIZE = (320, 320)
            print("[INFO] CPU mode: det_size auto-set to (320,320) for performance.")
        if ZOOM_CROP > 0:
            ZOOM_CROP = 0.0
            print("[INFO] CPU mode: ZOOM_CROP disabled (halves inference cost; enable on GPU).")
        if INFER_EVERY < 2:
            INFER_EVERY = 2
            print("[INFO] CPU mode: inference every 2nd frame (tracker fills skipped frames).")

    # Only load detection + recognition — skip 3D/2D landmark and gender/age models.
    # The detection model already provides the 5-point keypoints that ArcFace needs.
    app = FaceAnalysis(name="buffalo_l", providers=providers,
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=ctx_id, det_size=DET_SIZE)
    return app


# ---------------------------------------------------------------------------
# Quality filter (mirrors notebook cell 5b)
# ---------------------------------------------------------------------------
def face_sharpness(face, src_img: np.ndarray) -> float:
    crop = face_align.norm_crop(src_img, landmark=face.kps)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def quality_check(face, src_img: np.ndarray) -> bool:
    if face.det_score < MIN_DET_SCORE_RUNTIME:
        return False
    return face_sharpness(face, src_img) >= MIN_SHARPNESS_RUNTIME


# ---------------------------------------------------------------------------
# IoU + multi-scale detection + face tracker
# ---------------------------------------------------------------------------

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def _match_score(track_bbox: np.ndarray, det_bbox: np.ndarray) -> float:
    """IoU with centroid-distance fallback for fast-moving faces.

    When someone moves quickly the new bbox has low IoU with the old track,
    causing a stale ghost at the old position.  If the face centre moved less
    than 2 face-widths we treat it as the same person and return a small
    positive score so the track is updated rather than abandoned.
    """
    iou = _iou(track_bbox, det_bbox)
    if iou >= TRACK_MIN_IOU:
        return iou
    tc = (track_bbox[:2] + track_bbox[2:]) * 0.5
    dc = (det_bbox[:2]  + det_bbox[2:])  * 0.5
    dist = float(np.linalg.norm(tc - dc))
    fw   = float(det_bbox[2] - det_bbox[0])
    if fw > 0 and dist < fw * 2.0:
        return max(1e-9, TRACK_MIN_IOU * (1.0 - dist / (fw * 2.0)))
    return 0.0


def _nms_faces(faces: list, iou_thresh: float = 0.45) -> list:
    """NMS over face detections: keep highest-score detection when boxes overlap."""
    if len(faces) <= 1:
        return faces
    by_score = sorted(faces, key=lambda f: f.det_score, reverse=True)
    keep = []
    for f in by_score:
        if all(_iou(f.bbox, k.bbox) < iou_thresh for k in keep):
            keep.append(f)
    return keep


def detect_with_zoom(app: FaceAnalysis, frame: np.ndarray) -> list:
    """Full-frame detection + a zoomed center-crop pass for distant/small faces."""
    faces = list(app.get(frame))
    if not ZOOM_CROP or ZOOM_CROP >= 1.0:
        return _nms_faces(faces)

    h, w = frame.shape[:2]
    cx1 = int(w * (1 - ZOOM_CROP) / 2)
    cy1 = int(h * (1 - ZOOM_CROP) / 2)
    cx2 = int(w * (1 + ZOOM_CROP) / 2)
    cy2 = int(h * (1 + ZOOM_CROP) / 2)
    crop = frame[cy1:cy2, cx1:cx2]

    for f in app.get(crop):
        f.bbox[[0, 2]] += cx1
        f.bbox[[1, 3]] += cy1
        f.kps[:, 0]    += cx1
        f.kps[:, 1]    += cy1
        faces.append(f)

    # Single NMS pass over all detections from both passes to remove all duplicates
    return _nms_faces(faces)


class Track:
    _next_id = 0

    def __init__(self, bbox: np.ndarray, kps: np.ndarray, name: str, score: float):
        self.track_id  = Track._next_id
        Track._next_id += 1
        self.bbox      = bbox.copy()
        self.kps       = kps.copy()
        self.name      = name
        self.score     = score
        self.age       = 0           # frames since last matched (0 = matched this frame)
        self.last_seen = time.time() # wall-clock time of last match (for time-based expiry)


class FaceTracker:
    def __init__(self):
        self.tracks: list[Track] = []

    def update(
        self,
        faces: list,
        results: list[tuple[str, float]],
    ) -> list[Track]:
        now        = time.time()
        detections = list(zip(faces, results))
        matched_det: set[int] = set()

        # Increment age of all existing tracks; matched ones are reset to 0 below
        for t in self.tracks:
            t.age += 1

        # Greedy matching — IoU first, centroid-distance fallback for fast movement
        for track in self.tracks:
            best_score, best_idx = 0.0, -1
            for d_idx, (face, _) in enumerate(detections):
                if d_idx in matched_det:
                    continue
                v = _match_score(track.bbox, face.bbox)
                if v > best_score:
                    best_score, best_idx = v, d_idx

            if best_idx >= 0:
                face, (name, score) = detections[best_idx]
                track.bbox      = face.bbox.copy()
                track.kps       = face.kps.copy()
                track.age       = 0
                track.score     = score
                track.last_seen = now
                if track.name == "Unknown" and name != "Unknown":
                    track.name = name  # lock identity once known — prevents flickering
                matched_det.add(best_idx)

        # Unmatched detections → new tracks
        for d_idx, (face, (name, score)) in enumerate(detections):
            if d_idx not in matched_det:
                self.tracks.append(Track(face.bbox, face.kps, name, score))

        # Drop tracks not seen within TRACK_MAX_AGE seconds (time-based, FPS-independent)
        self.tracks = [t for t in self.tracks if now - t.last_seen <= TRACK_MAX_AGE]
        return self.tracks


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def l2(v: np.ndarray) -> np.ndarray:
    return v / norm(v)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def identify(embedding: np.ndarray, registry: dict) -> tuple[str, float]:
    if not registry:
        return "Unknown", 0.0
    scores = {name: cosine_sim(embedding, emb) for name, emb in registry.items()}
    best_name = max(scores, key=scores.get)
    best_score = scores[best_name]
    if best_score >= SIM_THRESHOLD:
        return best_name, best_score
    return "Unknown", best_score


# ---------------------------------------------------------------------------
# Registry — load from faces_db/ at startup
# ---------------------------------------------------------------------------

def _imread_exif(path) -> np.ndarray | None:
    """Read an image file respecting EXIF rotation.

    cv2.imread() ignores the EXIF orientation tag that phone cameras embed,
    so portrait photos saved as landscape raw data appear sideways to
    RetinaFace and fail detection.  PIL's exif_transpose() fixes this.
    """
    try:
        pil = ImageOps.exif_transpose(Image.open(str(path)))
        return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        return cv2.imread(str(path))  # fallback for non-JPEG or unreadable
def load_registry(app: FaceAnalysis) -> dict:
    registry = {}
    if not FACES_DB.exists():
        print(f"[WARN] {FACES_DB}/ not found — starting with empty registry.")
        return registry

    # Re-prepare with 640x640 for enrollment photos.  DET_SIZE (typically 1920x1920)
    # upscales close-up portraits so faces exceed RetinaFace's largest anchor (~512 px)
    # and go completely undetected.  640x640 is correct for close-up photos; ArcFace
    # embeddings are extracted on a normalised 112x112 crop so quality is unaffected.
    import onnxruntime as ort
    _ctx_id = 0 if "CUDAExecutionProvider" in ort.get_available_providers() else -1
    app.prepare(ctx_id=_ctx_id, det_size=(640, 640))

    for person_dir in sorted(FACES_DB.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        embeddings = []
        for img_path in sorted(person_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            img = _imread_exif(img_path)
            if img is None:
                print(f"    [SKIP] {img_path.name} — could not read file")
                continue
            faces = app.get(img)
            if not faces:
                print(f"    [SKIP] {img_path.name} — no face detected")
                continue
            if len(faces) > 1:
                print(f"    [WARN] {img_path.name} — {len(faces)} faces; "
                      f"using the largest (solo portrait photos give better accuracy)")
            # Take the largest face — in an enrollment photo the subject should be closest
            best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            if best.det_score < 0.20:
                print(f"    [SKIP] {img_path.name} — det_score {best.det_score:.2f} too low")
                continue
            sharp = face_sharpness(best, img)
            print(f"    [OK]   {img_path.name} — det={best.det_score:.2f}  sharp={sharp:.1f}  faces={len(faces)}")
            embeddings.append(l2(best.embedding))

        if embeddings:
            avg = np.mean(embeddings, axis=0)
            registry[name] = l2(avg)
            print(f"  Registered '{name}' from {len(embeddings)} photo(s)")
        else:
            print(f"  [WARN] No usable face found in {person_dir}/")

    print(f"[INFO] Registry loaded: {list(registry.keys()) or 'empty'}")

    # Restore the runtime detection size for the webcam loop
    app.prepare(ctx_id=_ctx_id, det_size=DET_SIZE)
    return registry


# ---------------------------------------------------------------------------
# Live enrollment
# ---------------------------------------------------------------------------
def enroll_live(
    name: str,
    cap: cv2.VideoCapture,
    app: FaceAnalysis,
    registry: dict,
) -> None:
    person_dir = FACES_DB / name
    person_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ENROLL] Capturing {ENROLL_FRAMES} frames for '{name}' — look at the camera…")
    collected: list[np.ndarray] = []
    saved = 0

    while len(collected) < ENROLL_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_with_zoom(app, frame)
        for f in faces:
            # Enrollment uses the strict thresholds regardless of runtime relaxation
            if f.det_score < MIN_DET_SCORE or face_sharpness(f, frame) < MIN_SHARPNESS:
                continue
            emb = l2(f.embedding)
            collected.append(emb)

            img_path = person_dir / f"webcam_{saved}.jpg"
            crop = face_align.norm_crop(frame, landmark=f.kps)
            cv2.imwrite(str(img_path), crop)
            saved += 1

            label = f"Capturing {len(collected)}/{ENROLL_FRAMES}"
            x1, y1, x2, y2 = f.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 8), FONT, 0.6, (255, 200, 0), 2)
            break

        cv2.imshow("InsightFace — Webcam Recognition", frame)
        cv2.waitKey(1)

    if collected:
        avg = np.mean(collected, axis=0)
        registry[name] = l2(avg)
        print(f"[ENROLL] '{name}' enrolled from {len(collected)} frame(s).")
    else:
        print(f"[ENROLL] No clear face captured for '{name}' — try again.")


# ---------------------------------------------------------------------------
# Frame annotation
# ---------------------------------------------------------------------------
def annotate_frame(frame: np.ndarray, tracks: list) -> None:
    for track in tracks:
        x1, y1, x2, y2 = track.bbox.astype(int)
        color     = COLOR_KNOWN if track.name != "Unknown" else COLOR_UNKNOWN
        thickness = 2 if track.age == 0 else 1  # thinner border for held/predicted tracks

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"{track.name}  {track.score:.2f}" if track.name != "Unknown" else "Unknown"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), FONT, 0.6, (255, 255, 255), 2)

        if track.age == 0:  # keypoints only when freshly detected; stale ones are inaccurate
            for kp in track.kps.astype(int):
                cv2.circle(frame, tuple(kp), 3, COLOR_KP, -1)


def draw_hud(frame: np.ndarray, registry: dict, fps: float) -> None:
    h, w = frame.shape[:2]
    lines = [
        f"FPS: {fps:.1f}",
        f"Registered: {', '.join(registry.keys()) or 'none'}",
        "r=enroll  q=quit",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, h - 15 - i * 22), FONT, 0.55, (200, 200, 200), 1)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
WIN_NAME = "InsightFace — Webcam Recognition"


def scale_to_window(frame: np.ndarray) -> np.ndarray:
    """Scale frame to DISPLAY_SIZE, preserving aspect ratio with black bars."""
    if DISPLAY_SIZE is None:
        return frame

    target_w, target_h = DISPLAY_SIZE
    h, w = frame.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if new_w == target_w and new_h == target_h:
        return resized

    # Pad with black bars to fill the target canvas
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time face recognition — webcam or RTSP stream",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/webcam_recognition.py                          # default webcam\n"
            "  python src/webcam_recognition.py --source 1               # webcam index 1\n"
            "  python src/webcam_recognition.py --source rtsp://192.168.1.10:554/stream\n"
        ),
    )
    parser.add_argument(
        "--source",
        default=None,
        metavar="SOURCE",
        help=(
            "Video source: camera index (0, 1, …) or RTSP URL (rtsp://…). "
            f"Defaults to webcam {CAMERA_INDEX}."
        ),
    )
    return parser.parse_args()


def _open_capture(source) -> cv2.VideoCapture:
    """Open a VideoCapture from a webcam index or RTSP/file URL."""
    is_rtsp = isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://"))

    if is_rtsp:
        # Use FFMPEG backend for RTSP; keep buffer small to minimise latency.
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[INFO] Connecting to RTSP stream: {source}")
    else:
        cap = cv2.VideoCapture(source)
        print(f"[INFO] Opening camera index {source}")

    return cap


def main() -> None:
    args = _parse_args()

    # Resolve the source: None → default camera index; digit string → int; else keep as URL/path
    if args.source is None:
        source = CAMERA_INDEX
    elif args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    print("[INFO] Initializing InsightFace (buffalo_l)…")
    app = build_app()
    registry = load_registry(app)

    cap = _open_capture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source!r}")
        sys.exit(1)

    # Create a resizable window so the user can drag its edges freely;
    # the initial size is set to DISPLAY_SIZE (or native camera res if None).
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    if DISPLAY_SIZE is not None:
        cv2.resizeWindow(WIN_NAME, DISPLAY_SIZE[0], DISPLAY_SIZE[1])

    print("[INFO] Stream open. Press 'r' to enroll, 'q' to quit.")

    tracker     = FaceTracker()
    fps         = 0.0
    prev_time   = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        # Run full inference every INFER_EVERY frames; tracker holds positions for skipped frames
        frame_count += 1
        if frame_count % INFER_EVERY == 0:
            faces   = detect_with_zoom(app, frame)
            clear   = [f for f in faces if quality_check(f, frame)]
            results = [identify(l2(f.embedding), registry) for f in clear]
            tracker.update(clear, results)

        annotate_frame(frame, tracker.tracks)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        draw_hud(frame, registry, fps)

        # Scale the annotated frame to the window size before displaying
        cv2.imshow(WIN_NAME, scale_to_window(frame))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            name = input("Enter name to enroll: ").strip()
            if name:
                enroll_live(name, cap, app, registry)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()