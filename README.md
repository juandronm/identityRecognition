# YourEye — Real-time Face Recognition

Real-time face recognition from a **webcam or RTSP camera stream** using **InsightFace** (RetinaFace + ArcFace).

---

## Requirements

- Python 3.10 or newer
- A webcam or an RTSP-capable IP camera
- Internet connection on the **first run** (downloads the `buffalo_l` model, ~400 MB)

---

## Installation

From the **project root** (`cwenerji_facedetection/`):

```bash
pip install -r requirements.txt
```

> **GPU vs CPU** — open `requirements.txt` and choose:
> - **NVIDIA GPU** → keep `onnxruntime-gpu` (also install the [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads))
> - **No GPU / any machine** → uncomment `onnxruntime` and comment out `onnxruntime-gpu`

---

## Enroll your face (before first run)

Create a folder named after yourself inside `faces_db/` and drop 1–3 clear, well-lit photos of your face in it:

```
faces_db/
  YourName/
    photo1.jpg
    photo2.jpg
```

The folder name becomes the identity label shown on screen.  
You can also enroll **live from the camera** during a session (press `r`, see controls below).

---

## Running

### Default webcam

```bash
python src/webcam_recognition.py
```

### Specific webcam index

```bash
python src/webcam_recognition.py --source 1
```

### RTSP camera stream

```bash
python src/webcam_recognition.py --source rtsp://192.168.1.10:554/stream
```

The script always finds `faces_db/` relative to its own location, regardless of where you launch it from.

---

## Keyboard controls

| Key | Action |
|-----|--------|
| `r` | Enroll a new face live — type a name in the terminal, then look at the camera |
| `q` | Quit |

---

## Tunables

Open `src/webcam_recognition.py` and adjust these constants at the top of the file:

| Constant | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `0` | Fallback webcam index when `--source` is not passed |
| `DISPLAY_SIZE` | `(1280, 720)` | Window size in pixels. Set to `None` for native camera resolution |
| `SIM_THRESHOLD` | `0.45` | Cosine similarity threshold — raise toward `0.55` for larger galleries |
| `MIN_DET_SCORE` | `0.60` | Minimum RetinaFace confidence for enrollment photos |
| `MIN_DET_SCORE_RUNTIME` | `0.35` | Minimum confidence at runtime (lower — catches distant/angled faces) |
| `MIN_SHARPNESS` | `50.0` | Laplacian variance floor for enrollment |
| `MIN_SHARPNESS_RUNTIME` | `10.0` | Sharpness floor at runtime |
| `ENROLL_FRAMES` | `5` | Frames captured per person during live enrollment |
| `TRACK_MAX_AGE` | `0.4` | Seconds to hold a track after a face disappears (keeps labels stable) |
| `INFER_EVERY` | `1` | Run full inference every N frames (auto-set to 2 on CPU for speed) |

---

## How it works

1. **Registry loading** — on startup, every photo in `faces_db/<Name>/` is processed through RetinaFace + ArcFace to produce a 512-dimensional embedding per person.
2. **Detection loop** — each frame is passed through RetinaFace; detected faces are quality-filtered and matched against the registry using cosine similarity.
3. **Tracking** — a lightweight IoU-based tracker holds each identity for up to `TRACK_MAX_AGE` seconds after detection, so labels stay on screen through brief occlusions or fast movement.
4. **Live enrollment** — press `r`, type a name, and the script captures `ENROLL_FRAMES` clean frames and adds them to the registry instantly.

GPU performance (CUDA): full `DET_SIZE=(1920,1920)` + zoom-crop pass for distant faces.  
CPU performance: auto-reduced to `DET_SIZE=(320,320)`, zoom disabled, inference every 2nd frame (~15–20 FPS).

---

## Project structure

```
cwenerji_facedetection/
├── faces_db/                  ← one subfolder per person, photos inside (git-ignored)
│   └── YourName/
│       └── photo.jpg
├── src/
│   └── webcam_recognition.py  ← main script
├── notebooks/
│   └── insigthFaceTrial.ipynb
└── requirements.txt
```
