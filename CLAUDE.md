# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
```

GPU (NVIDIA): keep `onnxruntime-gpu` in `requirements.txt` and install the CUDA Toolkit 12.x system-wide.  
CPU-only: swap to `onnxruntime` (comment/uncomment the two lines in `requirements.txt`).

The `buffalo_l` InsightFace model (~400 MB) downloads automatically on first run.
+
## Running

```bash
python src/webcam_recognition.py
```

The script resolves `faces_db/` relative to its own location, so it works from any working directory.

## Architecture

There is one production script: `src/webcam_recognition.py`. Everything else is exploratory (`notebooks/`) or static assets (`img/`).

**Data flow:**

1. `build_app()` — loads InsightFace (`buffalo_l`: RetinaFace detector + ArcFace R50 embedder) with CUDA or CPU fallback.
2. `load_registry()` — reads `faces_db/<Name>/*.jpg`, runs each photo through quality gates, extracts 512-d ArcFace embeddings, L2-normalises them, averages per person, stores one vector per identity in an in-memory `dict`.
3. Main loop — per frame: detect faces → `quality_check()` → `identify()` (cosine similarity against registry) → annotate → display.
4. Live enrollment (`enroll_live()`) — triggered by pressing `r`; captures `ENROLL_FRAMES` quality frames, writes crops to `faces_db/<Name>/`, updates the in-memory registry in place.

**Key tunables** (top of `webcam_recognition.py`):

| Constant | Default | Effect |
|---|---|---|
| `SIM_THRESHOLD` | 0.45 | Raise toward 0.55 for larger galleries (500+ people) to reduce false positives |
| `MIN_DET_SCORE` | 0.60 | Minimum RetinaFace confidence; raise to 0.75 for enrollment quality gates |
| `MIN_SHARPNESS` | 50.0 | Laplacian variance floor; raise to 80+ for enrollment |
| `ENROLL_FRAMES` | 5 | Frames captured during live enrollment; 10 recommended for production |
| `DET_SIZE` | (640,640) | Use (320,320) on CPU for better FPS |
| `CAMERA_INDEX` | 0 | Change if the target webcam is not the default device |

**Face database layout:**

```
faces_db/
  <PersonName>/        ← folder name becomes the identity label
    photo1.jpg
    photo2.jpg
    webcam_0.jpg       ← crops saved by live enrollment
```

`faces_db/` contents are fully git-ignored (see `faces_db/.gitignore`).

## CUDA Bootstrap

`_bootstrap_cuda()` runs before any ONNX import. It locates CUDA DLLs inside PyTorch's bundled `lib/` directory and pre-loads them so `onnxruntime-gpu`'s `CUDAExecutionProvider` can initialise without the CUDA Toolkit installed system-wide. If the Toolkit is present, this step is a no-op.
