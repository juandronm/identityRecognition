# webcam_recognition.py

Real-time face recognition from a webcam using **InsightFace** (RetinaFace + ArcFace).

---

## Requirements

- Python 3.10 or newer
- A webcam connected to your machine
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

Create a folder named after yourself inside `faces_db/` and drop 1–3 clear photos of your face in it:

```
faces_db/
  YourName/
    photo1.jpg
    photo2.jpg
```

The folder name becomes your identity label on screen.  
You can also enroll **live from the webcam** during a session (press `r`, see controls below).

---

## Running

From **any directory**:

```bash
python src/webcam_recognition.py
```

Or from inside `src/`:

```bash
python webcam_recognition.py
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

Open `webcam_recognition.py` and adjust these constants at the top of the file:

| Constant | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `0` | Webcam device index. Change to `1`, `2` … if your camera is not the default |
| `DISPLAY_SIZE` | `(1280, 720)` | Window size in pixels. Set to `None` for native camera resolution |
| `DET_SIZE` | `(640, 640)` | Detection input size. Use `(320, 320)` on CPU for better frame rate |
| `SIM_THRESHOLD` | `0.45` | Cosine similarity threshold — raise to be stricter, lower to be more lenient |
| `MIN_DET_SCORE` | `0.60` | Minimum RetinaFace confidence to accept a detection |
| `MIN_SHARPNESS` | `50.0` | Laplacian variance threshold — filters out blurry / distant faces |
| `ENROLL_FRAMES` | `5` | Number of frames captured per person during live enrollment |

---

## Project structure

```
cwenerji_facedetection/
├── faces_db/          ← one subfolder per person, photos inside
│   └── YourName/
│       └── photo.jpg
├── src/
│   ├── webcam_recognition.py
│   └── README.md      ← you are here
├── notebooks/
│   └── insigthFaceTrial.ipynb
└── requirements.txt
```
