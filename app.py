"""
Road Surface Damage Detection — Streamlit app.

Upload a road photo OR a video, pick one of the four trained YOLOv8s
variants (baseline / +CBAM / +P2 head / +CBAM+P2), and get back the
media annotated with detected cracks, potholes, and manholes.

Deploy on Streamlit Community Cloud: point it at this file as the
main entry point. See README.md for setup notes (esp. packages.txt).
"""
import os
import sys
import time
import tempfile
import warnings

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# --- Make CBAM classes resolvable under __main__ before anything loads a
# checkpoint that was pickled with them (see cbam_module.py for why). ---
from cbam_module import CBAM, ChannelAttention, SpatialAttention, register_cbam

sys.modules["__main__"].CBAM = CBAM
sys.modules["__main__"].ChannelAttention = ChannelAttention
sys.modules["__main__"].SpatialAttention = SpatialAttention
register_cbam()

from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_REGISTRY = {
    "baseline": {
        "label": "YOLOv8s (baseline)",
        "file": "yolov8s_baseline.pt",
        "desc": "Standard YOLOv8s detector, no architecture changes.",
        "tag": "Fastest",
    },
    "cbam": {
        "label": "YOLOv8s + CBAM",
        "file": "yolov8s_cbam.pt",
        "desc": "Adds Convolutional Block Attention Modules to the backbone.",
        "tag": "Attention-based",
    },
    "p2": {
        "label": "YOLOv8s + P2 head",
        "file": "yolov8s_p2.pt",
        "desc": "Adds a high-resolution P2 detection head for small objects (e.g. thin cracks).",
        "tag": "Best for small damage",
    },
    "cbam_p2": {
        "label": "YOLOv8s + CBAM + P2",
        "file": "yolov8s_cbam_p2.pt",
        "desc": "Combines the CBAM attention blocks with the P2 detection head.",
        "tag": "Most complex",
    },
}

CLASS_COLORS_BGR = {
    "crack": (245, 135, 66),      # blue-ish
    "pothole": (52, 64, 235),     # red-ish
    "manhole": (89, 199, 52),     # green-ish
}
CLASS_COLORS_HEX = {"crack": "#4287f5", "pothole": "#eb4034", "manhole": "#34c759"}

st.set_page_config(
    page_title="Road Surface Damage Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------ THEME / CSS ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0f1115; }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #1a1d24 0%, #23272f 100%);
        border: 1px solid #2e333d;
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
    }
    .hero h1 {
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        color: #f5f5f7;
        letter-spacing: -0.02em;
    }
    .hero p {
        color: #9aa0a8;
        font-size: 0.98rem;
        margin: 0;
    }
    .hero .pill {
        display: inline-block;
        background: #f2c14e;
        color: #17191c;
        font-weight: 700;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 20px;
        margin-bottom: 12px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #15171c;
        border-right: 1px solid #2e333d;
    }
    section[data-testid="stSidebar"] * {
        color: #05c1c5 !important;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9aa0a8 !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #8a8f98 !important;
    }

    /* Cards */
    .info-card {
        background: #1a1d24;
        border: 1px solid #2e333d;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .model-tag {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 2px 9px;
        border-radius: 12px;
        background: rgba(242, 193, 78, 0.15);
        color: #f2c14e;
        margin-bottom: 6px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1d24;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        color: #9aa0a8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #23272f;
        color: #f2c14e !important;
        border-bottom: 2px solid #f2c14e;
    }

    /* Buttons */
    div[data-testid="stButton"] button[kind="primary"] {
        background: #f2c14e;
        color: #17191c;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 10px 0;
        transition: filter 0.15s;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover { filter: brightness(1.08); }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #1a1d24;
        border: 1px solid #2e333d;
        border-radius: 10px;
        padding: 14px 16px 10px;
    }
    div[data-testid="stMetricLabel"] { color: #9aa0a8 !important; }
    div[data-testid="stMetricLabel"] * { color: #9aa0a8 !important; }
    div[data-testid="stMetricValue"] { color: #f5f5f7 !important; }
    div[data-testid="stMetricValue"] * { color: #f5f5f7 !important; }

    /* Result caption line */
    .result-meta {
        color: #9aa0a8;
        font-size: 0.85rem;
        margin: 4px 0 16px 0;
        font-family: 'Courier New', monospace;
    }

    /* Legend chips */
    .legend { display: flex; gap: 14px; margin: 6px 0 18px 0; flex-wrap: wrap; }
    .legend .chip {
        display: flex; align-items: center; gap: 6px;
        font-size: 0.82rem; color: rgb(49, 51, 63); !important;
    }
    .legend .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

    footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model(model_id: str) -> YOLO:
    weight_path = os.path.join(MODELS_DIR, MODEL_REGISTRY[model_id]["file"])
    return YOLO(weight_path)


def draw_detections(frame_bgr, detections):
    img = frame_bgr.copy()
    h = img.shape[0]
    thickness = max(2, round(h / 500))
    font_scale = max(0.5, h / 1400)
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color = CLASS_COLORS_BGR.get(d["class"], (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        label = f'{d["class"]} {d["confidence"]:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        ty1 = max(0, y1 - th - 8)
        cv2.rectangle(img, (x1, ty1), (x1 + tw + 6, ty1 + th + 8), color, -1)
        cv2.putText(img, label, (x1 + 3, ty1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def run_on_frame(model, frame_bgr, conf, imgsz):
    results = model.predict(source=frame_bgr, conf=conf, imgsz=imgsz, verbose=False)
    r = results[0]
    detections = []
    for box in r.boxes:
        cls_id = int(box.cls.item())
        cls_name = model.names[cls_id]
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        detections.append({
            "class": cls_name,
            "confidence": confidence,
            "box": (x1, y1, x2, y2),
        })
    return detections


def process_video(model, in_path, out_path, conf, imgsz, frame_skip, progress_cb=None):
    """Run detection on a video, writing an annotated copy to out_path.
    Frames are processed every `frame_skip`-th frame; boxes from the last
    processed frame are re-drawn on skipped frames so playback stays smooth.
    Returns aggregate class counts (unique boxes counted per processed frame).
    """
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    counts = {}
    last_detections = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_skip == 0:
            last_detections = run_on_frame(model, frame, conf, imgsz)
            for d in last_detections:
                counts[d["class"]] = counts.get(d["class"], 0) + 1

        annotated = draw_detections(frame, last_detections)
        writer.write(annotated)

        frame_idx += 1
        if progress_cb and total_frames:
            progress_cb(min(frame_idx / total_frames, 1.0))

    cap.release()
    writer.release()
    return counts, frame_idx, fps


def reencode_for_browser(in_path, out_path):
    """Re-mux/encode with imageio-ffmpeg (H.264) so the result reliably
    plays back in-browser via st.video — OpenCV's mp4v fourcc often
    doesn't. Falls back to the original file if ffmpeg isn't available."""
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        import subprocess
        subprocess.run(
            [ffmpeg_exe, "-y", "-i", in_path, "-vcodec", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "23", out_path],
            check=True, capture_output=True,
        )
        return out_path
    except Exception:
        return in_path


def render_legend():
    chips = "".join(
        f'<div class="chip"><span class="dot" style="background:{hexcol}"></span>{name}</div>'
        for name, hexcol in CLASS_COLORS_HEX.items()
    )
    st.markdown(f'<div class="legend">{chips}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- UI ----

st.markdown("""
<div class="hero">
    <div class="pill">YOLOv8s · 4 architectures</div>
    <h1>🛣️ Road Surface Damage Detection</h1>
    <p>Upload a photo or video and detect cracks, potholes, and manholes — compare four trained model variants side by side.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Model")
    model_id = st.selectbox(
        "Choose architecture",
        options=list(MODEL_REGISTRY.keys()),
        format_func=lambda k: MODEL_REGISTRY[k]["label"],
        label_visibility="collapsed",
    )
    info = MODEL_REGISTRY[model_id]
    st.markdown(f"""
    <div class="info-card">
        <span class="model-tag">{info['tag']}</span>
        <div style="font-weight:600; margin-bottom:4px; color:#f5f5f7;">{info['label']}</div>
        <div style="color:#9aa0a8; font-size:0.85rem;">{info['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.header("Detection settings")
    conf = st.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)
    imgsz = st.select_slider("Inference size (px)", options=[512, 640, 768, 1024], value=768)

    st.divider()
    st.caption(
        "💡 Lower inference size and higher frame-skip make video "
        "processing much faster on CPU-only deployments."
    )

tab_image, tab_video, tab_compare = st.tabs(
    ["🖼️  Image", "🎥  Video", "⚖️  Compare Models"]
)

# ---- Image tab ----
with tab_image:
    img_file = st.file_uploader(
        "Upload a road photo", type=["jpg", "jpeg", "png", "bmp", "webp"], key="img_uploader"
    )
    if img_file is not None:
        file_bytes = np.frombuffer(img_file.read(), np.uint8)
        frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)

        if st.button("Run detection", type="primary", key="run_img", use_container_width=True):
            with st.spinner(f"Running {MODEL_REGISTRY[model_id]['label']}…"):
                model = load_model(model_id)
                t0 = time.time()
                detections = run_on_frame(model, frame_bgr, conf, imgsz)
                elapsed = round(time.time() - t0, 3)
                annotated = draw_detections(frame_bgr, detections)

            with col2:
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detections", use_container_width=True)

            st.markdown(
                f'<div class="result-meta">{MODEL_REGISTRY[model_id]["label"]} · '
                f'inference {elapsed}s · {len(detections)} detection(s)</div>',
                unsafe_allow_html=True,
            )
            render_legend()

            counts = {}
            for d in detections:
                counts[d["class"]] = counts.get(d["class"], 0) + 1

            c1, c2, c3 = st.columns(3)
            c1.metric("🔵 Cracks", counts.get("crack", 0))
            c2.metric("🔴 Potholes", counts.get("pothole", 0))
            c3.metric("🟢 Manholes", counts.get("manhole", 0))

            st.write("")
            if detections:
                st.dataframe(
                    [{"class": d["class"], "confidence": round(d["confidence"], 3), "box (x1,y1,x2,y2)": d["box"]}
                     for d in sorted(detections, key=lambda d: -d["confidence"])],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No damage detected above the confidence threshold.")

            ok, buf = cv2.imencode(".jpg", annotated)
            if ok:
                st.download_button(
                    "⬇ Download annotated image", data=buf.tobytes(),
                    file_name="detection_result.jpg", mime="image/jpeg",
                    use_container_width=True,
                )

# ---- Video tab ----
with tab_video:
    st.caption(
        "Detection runs frame by frame — this can take a while on CPU. "
        "Use the frame-skip control below to trade speed for smoothness."
    )
    vid_file = st.file_uploader(
        "Upload a road video", type=["mp4", "mov", "avi", "mkv"], key="vid_uploader"
    )
    frame_skip = st.slider(
        "Process every Nth frame (higher = faster)", 1, 15, 3, 1, key="frame_skip"
    )

    if vid_file is not None:
        st.video(vid_file)

        if st.button("Run detection on video", type="primary", key="run_vid", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(vid_file.name)[1]) as tmp_in:
                tmp_in.write(vid_file.read())
                in_path = tmp_in.name

            raw_out_path = in_path + "_annotated_raw.mp4"
            final_out_path = in_path + "_annotated.mp4"

            model = load_model(model_id)
            progress_bar = st.progress(0.0, text="Processing frames…")

            try:
                t0 = time.time()
                counts, n_frames, fps = process_video(
                    model, in_path, raw_out_path, conf, imgsz, frame_skip,
                    progress_cb=lambda p: progress_bar.progress(p, text=f"Processing frames… {int(p*100)}%"),
                )
                elapsed = round(time.time() - t0, 1)
                progress_bar.progress(1.0, text="Re-encoding for playback…")
                final_path = reencode_for_browser(raw_out_path, final_out_path)
                progress_bar.empty()

                st.success(f"Done — {n_frames} frames at {fps:.1f} fps processed in {elapsed}s")
                render_legend()

                c1, c2, c3 = st.columns(3)
                c1.metric("🔵 Crack frames", counts.get("crack", 0))
                c2.metric("🔴 Pothole frames", counts.get("pothole", 0))
                c3.metric("🟢 Manhole frames", counts.get("manhole", 0))
                st.caption("Counts are per processed frame (a persisting object across frames is counted each time it's re-detected), not unique objects.")

                st.video(final_path)
                with open(final_path, "rb") as f:
                    st.download_button(
                        "⬇ Download annotated video", data=f.read(),
                        file_name="detection_result.mp4", mime="video/mp4",
                        use_container_width=True,
                    )
            except Exception as e:
                progress_bar.empty()
                st.error(f"Video processing failed: {e}")
            finally:
                for p in (in_path, raw_out_path):
                    if os.path.exists(p) and p != final_out_path:
                        try:
                            os.remove(p)
                        except OSError:
                            pass

# ---- Compare tab ----
with tab_compare:
    st.caption(
        "Run the same image through all four trained architectures at once "
        "to see how they differ on the same input."
    )
    cmp_file = st.file_uploader(
        "Upload a road photo", type=["jpg", "jpeg", "png", "bmp", "webp"], key="cmp_uploader"
    )

    if cmp_file is not None:
        file_bytes = np.frombuffer(cmp_file.read(), np.uint8)
        frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), caption="Original", width=420)

        if st.button("Compare all 4 models", type="primary", key="run_cmp", use_container_width=True):
            render_legend()
            results_summary = []
            cols = st.columns(2)

            for i, mid in enumerate(MODEL_REGISTRY.keys()):
                with cols[i % 2]:
                    with st.spinner(f"Running {MODEL_REGISTRY[mid]['label']}…"):
                        model = load_model(mid)
                        t0 = time.time()
                        detections = run_on_frame(model, frame_bgr, conf, imgsz)
                        elapsed = round(time.time() - t0, 3)
                        annotated = draw_detections(frame_bgr, detections)

                    counts = {}
                    for d in detections:
                        counts[d["class"]] = counts.get(d["class"], 0) + 1

                    st.markdown(f"""
                    <div class="info-card" style="margin-top:12px;">
                        <span class="model-tag">{MODEL_REGISTRY[mid]['tag']}</span>
                        <div style="font-weight:700; color:#f5f5f7; margin-bottom:2px;">{MODEL_REGISTRY[mid]['label']}</div>
                        <div style="color:#9aa0a8; font-size:0.8rem;">{elapsed}s · {len(detections)} detection(s)</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

                    results_summary.append({
                        "Model": MODEL_REGISTRY[mid]["label"],
                        "Inference (s)": elapsed,
                        "Cracks": counts.get("crack", 0),
                        "Potholes": counts.get("pothole", 0),
                        "Manholes": counts.get("manhole", 0),
                        "Total": len(detections),
                    })

            st.write("")
            st.subheader("Summary")
            st.dataframe(results_summary, use_container_width=True, hide_index=True)
