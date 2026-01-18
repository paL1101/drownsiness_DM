import time
import argparse
import cv2
import numpy as np
import joblib
import mediapipe as mp

# Pakai config & accumulator yang sudah ada di project Anda
from config import WINDOW_SECONDS, EAR_CLOSED_THRESHOLD
from features import combined_ear, WindowAccumulator

FEATURES = [
    "ear_mean", "ear_min", "ear_std",
    "perclos", "blink_count",
    "avg_closure_ms", "max_closure_ms"
]

def open_camera(preferred_index: int, try_indices=(0, 1, 2, 3, 4, 5)):
    indices = [preferred_index] + [i for i in try_indices if i != preferred_index]
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # Windows: lebih stabil
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"[OK] Camera opened at index {idx}")
                return cap, idx
        try:
            cap.release()
        except:
            pass
    return None, None

def draw_label(img, lines, x=12, y=24):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2

    sizes = [cv2.getTextSize(t, font, scale, thickness)[0] for t in lines]
    w = max(s[0] for s in sizes) + 18
    h = (len(lines) * 28) + 14

    # background box gelap
    cv2.rectangle(img, (x-8, y-22), (x-8+w, y-22+h), (0, 0, 0), -1)

    yy = y
    for t in lines:
        # outline putih
        cv2.putText(img, t, (x, yy), font, scale, (255,255,255), thickness+2, cv2.LINE_AA)
        # teks hitam di atasnya (kontras tinggi)
        cv2.putText(img, t, (x, yy), font, scale, (0,0,0), thickness, cv2.LINE_AA)
        yy += 28

def draw_hud(img_bgr, lines, x=12, y=28, font_scale=0.7):
    """
    HUD kontras tinggi:
    - box hitam di belakang teks
    - outline putih + teks hitam (tetap terbaca saat lampu terang)
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    line_h = int(28 * font_scale / 0.7)

    # ukur box
    sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t in lines]
    box_w = max(s[0] for s in sizes) + 20
    box_h = (len(lines) * line_h) + 18

    # box hitam
    cv2.rectangle(img_bgr, (x - 10, y - 22), (x - 10 + box_w, y - 22 + box_h), (0, 0, 0), -1)

    yy = y
    for t in lines:
        # teks tetep putih
        cv2.putText(img_bgr, t, (x, yy), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        yy += line_h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/rf_model.pkl")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--task_model", type=str, default="models/mediapipe/face_landmarker.task")
    args = parser.parse_args()

    # Load model klasifikasi (RF)
    model = joblib.load(args.model)

    # Open camera
    cap, used_idx = open_camera(args.camera)
    if cap is None:
        raise RuntimeError(
            "Tidak bisa membuka kamera.\n"
            "Solusi cepat:\n"
            "1) Tutup Zoom/Meet/Discord/OBS/Camera app.\n"
            "2) Windows Settings > Privacy & security > Camera: ON untuk desktop apps.\n"
            "3) Coba index lain: --camera 0/1/2\n"
        )

    # --- MediaPipe Tasks: FaceLandmarker (VIDEO mode) ---
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.task_model),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    acc = WindowAccumulator(EAR_CLOSED_THRESHOLD)
    window_start = time.time()

    last_pred = None
    last_prob = 0.0

    # timestamp untuk detect_for_video harus naik (ms)
    start_ts = time.time()

    print("Realtime app (Tasks) berjalan. Tekan 'q' untuk keluar.")

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_bgr = cv2.flip(frame_bgr, 1)
            h, w = frame_bgr.shape[:2]

            # MediaPipe Image expects SRGB (RGB)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int((time.time() - start_ts) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            ear_val = None
            if result.face_landmarks and len(result.face_landmarks) > 0:
                # face_landmarks[0] berisi 478 normalized landmarks (x,y,z)
                lm = result.face_landmarks[0]
                pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)

                # combined_ear() dari features.py (harus sesuai indeks yang dipakai di project Anda)
                ear_val = float(combined_ear(pts))

                t_rel = time.time() - window_start
                acc.add(ear_val, t_rel)

            elapsed = time.time() - window_start

            # Prediksi per window
            if elapsed >= WINDOW_SECONDS:
                feats = acc.finalize(WINDOW_SECONDS)
                if feats is not None:
                    x = np.array([[feats[f] for f in FEATURES]], dtype=np.float32)

                    if hasattr(model, "predict_proba"):
                        last_prob = float(model.predict_proba(x)[0][1])
                    else:
                        last_prob = 0.0

                    last_pred = int(model.predict(x)[0])

                    status = "NGANTUK" if last_pred == 1 else "TIDAK NGANTUK"
                    print(f"[PRED] {status} | P(ngantuk)={last_prob:.3f} | feats={feats}")

                window_start = time.time()

            # Overlay
            status = "—"
            if last_pred is not None:
                status = "NGANTUK" if last_pred == 1 else "TIDAK NGANTUK"

            lines = [
                f"Cam idx: {used_idx}",
                f"Window: {elapsed:.1f}/{WINDOW_SECONDS:.0f}s",
                f"EAR: {ear_val:.3f} (th={EAR_CLOSED_THRESHOLD:.2f})" if ear_val is not None else "EAR: Face not detected",
                f"Status: {status}  P={last_prob:.2f}" if last_pred is not None else "Status: (waiting...)",
            ]

            draw_hud(frame_bgr, lines)

            cv2.imshow("Drowsiness Detection (RF + FaceLandmarker Tasks)", frame_bgr)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
