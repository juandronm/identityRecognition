"""
Run this from the project root:
    python src/diagnose_registry.py

It tests every photo in faces_db/ with both cv2 and PIL+EXIF reading,
and reports exactly what RetinaFace finds (or doesn't find) in each one.
"""
from pathlib import Path
import cv2
import numpy as np

try:
    from PIL import Image, ImageOps
    PIL_OK = True
except ImportError:
    PIL_OK = False
    print("[WARN] Pillow not installed — skipping PIL+EXIF test")

from insightface.app import FaceAnalysis

FACES_DB = Path(__file__).parent.parent / "faces_db"


def sharpness(img: np.ndarray, bbox) -> float:
    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


print("[INFO] Phase 1 — det_size=640 (same as diagnostic)")
app640 = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app640.prepare(ctx_id=-1, det_size=(640, 640))
print()
print("[INFO] Phase 2 — det_size=1920 (same as main script on GPU)")
app1920 = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app1920.prepare(ctx_id=-1, det_size=(1920, 1920))
print()

for person_dir in sorted(FACES_DB.iterdir()):
    if not person_dir.is_dir():
        continue
    print(f"{'='*60}")
    print(f"  Person: {person_dir.name}")
    print(f"{'='*60}")

    for img_path in sorted(person_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue

        print(f"\n  File: {img_path.name}")

        if PIL_OK:
            try:
                pil = ImageOps.exif_transpose(Image.open(str(img_path)))
                img = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
            except Exception as e:
                img = cv2.imread(str(img_path))
                print(f"    [PIL fallback] {e}")
        else:
            img = cv2.imread(str(img_path))

        if img is None:
            print(f"    CANNOT READ FILE")
            continue

        h, w = img.shape[:2]

        for label, app in [("det=640 ", app640), ("det=1920", app1920)]:
            faces = app.get(img)
            if not faces:
                print(f"    [{label}] {w}x{h}: NO FACE DETECTED")
            else:
                best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                bw = int(best.bbox[2] - best.bbox[0])
                bh = int(best.bbox[3] - best.bbox[1])
                sharp = sharpness(img, best.bbox)
                flag = "  ← BLURRY (old gate=50)" if sharp < 50 else ""
                print(f"    [{label}] {w}x{h}: det={best.det_score:.3f}  bbox={bw}x{bh}  sharp={sharp:.1f}{flag}")

    print()

print("Done.")
print()
print("Look for lines with 'NO FACE DETECTED' or '← BLURRY' — those explain why a person is skipped.")
