# Road Surface Damage Detection — Streamlit App

Upload a road **photo or video**, pick one of four trained YOLOv8s
variants, and get back the media annotated with detected **cracks**,
**potholes**, and **manholes**.

## What's included

- `app.py` — the whole app (image tab + video tab)
- `cbam_module.py` — custom CBAM attention layer definitions from the training notebooks (needed to load the CBAM checkpoints)
- `models/*.pt` — the four trained checkpoints:
  - `yolov8s_baseline.pt` — plain YOLOv8s
  - `yolov8s_p2.pt` — YOLOv8s + extra P2 detection head (better on small objects)
  - `yolov8s_cbam.pt` — YOLOv8s + CBAM attention blocks
  - `yolov8s_cbam_p2.pt` — both combined
- `requirements.txt` — Python deps
- `packages.txt` — system (apt) deps Streamlit Cloud installs automatically (`libgl1`, `libglib2.0-0` — OpenCV needs these even in headless mode)

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Point it at your repo, branch, and set **Main file path** to `app.py`.
4. Deploy. Streamlit Cloud reads `requirements.txt` and `packages.txt` automatically — no extra config needed.

**A few things worth knowing before you deploy:**

- **Repo size.** The four `.pt` files total ~90MB. GitHub's per-file limit is 100MB, so a normal `git add`/`git push` works fine — you don't need Git LFS. If you later add bigger models, you would.
- **Free tier is CPU-only with ~1GB RAM.** Image detection is fast (roughly a second or two per photo). Video is the expensive part — it runs the model frame by frame. The app exposes a **"process every Nth frame"** slider and an **inference size** control specifically so you (or your users) can trade quality for speed; defaults are tuned to be reasonable on free-tier hardware, but a long/high-res video can still be slow or hit the memory ceiling. Consider capping upload size or suggesting short clips if that becomes an issue.
- **Video playback.** OpenCV's `mp4v` writer isn't reliably playable in-browser, so after processing, the app re-encodes the result to H.264 using `imageio-ffmpeg` (a bundled static ffmpeg binary — no system ffmpeg install required, works on Streamlit Cloud out of the box).
- **Model caching.** Each model is loaded once and cached via `st.cache_resource`, so switching back to a previously-used model is instant; the first use of any given model still takes a couple seconds.

## Using the app

- **Image tab:** upload a jpg/png/webp, hit *Run detection*, get the annotated image, a per-class count, a detections table, and a download button.
- **Video tab:** upload an mp4/mov/avi/mkv, set the frame-skip, hit *Run detection on video*, watch the progress bar, then preview and download the annotated result.

## Notes on the CBAM models

Two of the four checkpoints were trained with a custom CBAM
(Convolutional Block Attention Module) layer that isn't part of stock
Ultralytics — it was defined inline in the training notebooks and
monkey-patched into `ultralytics.nn.tasks`. `cbam_module.py` reproduces
those exact class definitions and registers them the same way so the
saved weights load correctly. This is required infrastructure, not
optional — don't remove it even if you only plan to use the baseline
or P2 model, since it runs once at import time regardless.
