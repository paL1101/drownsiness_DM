import os
import time
import numpy as np
import cv2
import joblib
import gradio as gr
import mediapipe as mp

# gunakan modul Anda
from src.config import WINDOW_SECONDS, EAR_CLOSED_THRESHOLD
from src.features import combined_ear, WindowAccumulator

FEATURES = [
    "ear_mean", "ear_min", "ear_std",
    "perclos", "blink_count",
    "avg_closure_ms", "max_closure_ms"
]

RF_MODEL_PATH = os.path.join("models", "rf_model.pkl")
TASK_MODEL_PATH = os.path.join("models", "mediapipe", "face_landmarker.task")

# --- MediaPipe Tasks FaceLandmarker (VIDEO) ---
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

def create_landmarker():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=TASK_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)

def draw_label(img_bgr, lines, x=12, y=18):
    """
    Agar terbaca di kondisi terang:
    - bikin box gelap di belakang teks
    - teks warna hitam + outline putih (atau sebaliknya)
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2

    # hitung ukuran box
    sizes = [cv2.getTextSize(t, font, scale, thickness)[0] for t in lines]
    w = max(s[0] for s in sizes) + 18
    h = (len(lines) * 24) + 14

    # box gelap semi-solid (di OpenCV tidak native alpha, jadi solid cukup)
    cv2.rectangle(img_bgr, (x-8, y-14), (x-8+w, y-14+h), (0, 0, 0), -1)

    yy = y
    for t in lines:
        # outline putih dulu
        cv2.putText(img_bgr, t, (x, yy), font, scale, (255, 255, 255), thickness+2, cv2.LINE_AA)
        # teks hitam di atasnya
        cv2.putText(img_bgr, t, (x, yy), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
        yy += 24

    return img_bgr

def make_demo():
    rf = joblib.load(RF_MODEL_PATH)
    landmarker = create_landmarker()

    acc = WindowAccumulator(EAR_CLOSED_THRESHOLD)
    window_start = time.time()
    start_ts = time.time()

    last_pred = None
    last_prob = 0.0
    last_ear = None

    def infer(frame_rgb):
        nonlocal window_start, start_ts, last_pred, last_prob, last_ear, acc

        if frame_rgb is None:
            return None, "No frame"

        # gradio memberi RGB (H,W,3)
        h, w = frame_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int((time.time() - start_ts) * 1000)
        res = landmarker.detect_for_video(mp_image, timestamp_ms)

        ear_val = None
        if res.face_landmarks and len(res.face_landmarks) > 0:
            lm = res.face_landmarks[0]
            pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
            ear_val = float(combined_ear(pts))
            acc.add(ear_val, time.time() - window_start)

        elapsed = time.time() - window_start

        if elapsed >= WINDOW_SECONDS:
            feats = acc.finalize(WINDOW_SECONDS)
            if feats is not None:
                x = np.array([[feats[f] for f in FEATURES]], dtype=np.float32)
                if hasattr(rf, "predict_proba"):
                    last_prob = float(rf.predict_proba(x)[0][1])
                else:
                    last_prob = 0.0
                last_pred = int(rf.predict(x)[0])
            window_start = time.time()

        last_ear = ear_val

        # render overlay
        img_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        status = "—"
        if last_pred is not None:
            status = "NGANTUK" if last_pred == 1 else "TIDAK NGANTUK"

        lines = [
            f"Status: {status}   P(ngantuk): {last_prob:.2f}",
            f"EAR: {last_ear:.3f}" if last_ear is not None else "EAR: (face not detected)",
            f"Window: {min(elapsed, WINDOW_SECONDS):.1f}/{WINDOW_SECONDS:.0f}s"
        ]
        img_bgr = draw_label(img_bgr, lines)

        img_out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_out, f"{status} | P={last_prob:.2f}"

    return infer

CSS = """
/* High contrast supaya terbaca di tempat terang */
:root { color-scheme: light; }
.gradio-container { background: #ffffff !important; }
h1,h2,h3,p,span,label,div { color: #111111 !important; }
"""

with gr.Blocks(css=CSS, title="Drowsiness Detection (RF + FaceLandmarker)") as demo:
    gr.Markdown("# Drowsiness Detection (Dataset-trained RF + Webcam)\nGunakan kamera browser untuk pembuktian realtime.")

    with gr.Row():
        with gr.Column():
            cam = gr.Image(sources=["webcam"], streaming=True, label="Webcam (Live)")
        with gr.Column():
            out_img = gr.Image(label="Output (Overlay)")
            out_txt = gr.Textbox(label="Hasil")

    infer_fn = make_demo()
    cam.stream(infer_fn, inputs=cam, outputs=[out_img, out_txt])

demo.queue().launch()
