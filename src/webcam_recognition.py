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
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ---------------------------------------------------------------------------
# Tunables — adjust these without touching the rest of the script
# ---------------------------------------------------------------------------
FACES_DB      = _HERE.parent / "faces_db"   # sibling of src/ — safe regardless of cwd
CAMERA_INDEX  = 0            # change to 1, 2 … if your webcam is not the default device
DET_SIZE      = (640, 640)   # use (320, 320) on CPU-only for better fps
MIN_DET_SCORE = 0.60
MIN_SHARPNESS = 50.0
SIM_THRESHOLD = 0.45
ENROLL_FRAMES = 5            # good-quality frames to capture during live enrollment

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
        print("[INFO] Falling back to CPU — consider (320, 320) det_size for better fps.")
    else:
        print("[INFO] No GPU detected — running on CPU.")

    app = FaceAnalysis(name="buffalo_l", providers=providers)
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
    if face.det_score < MIN_DET_SCORE:
        return False
    return face_sharpness(face, src_img) >= MIN_SHARPNESS


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
def load_registry(app: FaceAnalysis) -> dict:
    registry = {}
    if not FACES_DB.exists():
        print(f"[WARN] {FACES_DB}/ not found — starting with empty registry.")
        return registry

    for person_dir in sorted(FACES_DB.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        embeddings = []
        for img_path in sorted(person_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            faces = app.get(img)
            for f in faces:
                if quality_check(f, img):
                    embeddings.append(l2(f.embedding))
                    break  # one face per enrollment photo is enough

        if embeddings:
            avg = np.mean(embeddings, axis=0)
            registry[name] = l2(avg)
            print(f"  Registered '{name}' from {len(embeddings)} photo(s)")
        else:
            print(f"  [WARN] No usable face found in {person_dir}/")

    print(f"[INFO] Registry loaded: {list(registry.keys()) or 'empty'}")
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

        faces = app.get(frame)
        for f in faces:
            if quality_check(f, frame):
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
def annotate_frame(
    frame: np.ndarray,
    faces: list,
    results: list[tuple[str, float]],
) -> None:
    for face, (name, score) in zip(faces, results):
        x1, y1, x2, y2 = face.bbox.astype(int)
        color = COLOR_KNOWN if name != "Unknown" else COLOR_UNKNOWN

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{name}  {score:.2f}" if name != "Unknown" else "Unknown"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), FONT, 0.6, (255, 255, 255), 2)

        for kp in face.kps.astype(int):
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
def main() -> None:
    print("[INFO] Initializing InsightFace (buffalo_l)…")
    app = build_app()
    registry = load_registry(app)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam (device 0).")
        sys.exit(1)

    # Create a resizable window so the user can drag its edges freely;
    # the initial size is set to DISPLAY_SIZE (or native camera res if None).
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    if DISPLAY_SIZE is not None:
        cv2.resizeWindow(WIN_NAME, DISPLAY_SIZE[0], DISPLAY_SIZE[1])

    print("[INFO] Webcam open. Press 'r' to enroll, 'q' to quit.")

    fps = 0.0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        # Inference runs on the full-resolution camera frame for best accuracy
        faces = app.get(frame)
        clear = [f for f in faces if quality_check(f, frame)]
        results = [identify(l2(f.embedding), registry) for f in clear]

        annotate_frame(frame, clear, results)

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
