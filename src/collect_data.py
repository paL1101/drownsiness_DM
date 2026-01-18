# src/collect_data.py
import os
import time
import argparse
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from config import WINDOW_SECONDS, EAR_CLOSED_THRESHOLD
from features import combined_ear, WindowAccumulator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=int, required=True, choices=[0, 1],
                        help="0=tidak ngantuk, 1=ngantuk (simulasi)")
    parser.add_argument("--out", type=str, default="data/drowsiness_features.csv",
                        help="output CSV path")
    parser.add_argument("--camera", type=int, default=0, help="kamera index")
    parser.add_argument("--windows", type=int, default=12,
                        help="jumlah window 10 detik yang ingin direkam (12=2 menit)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Tidak bisa membuka kamera. Coba ganti --camera 1 atau cek izin kamera.")

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    rows = []
    acc = WindowAccumulator(EAR_CLOSED_THRESHOLD)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    window_id = 0
    t0 = time.time()
    w_start = t0

    print("\nInstruksi:")
    print("- Tekan 'q' untuk berhenti.")
    print(f"- Merekam {args.windows} window, masing-masing {WINDOW_SECONDS:.0f} detik.")
    print(f"- Label sesi = {args.label}\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        ear_val = None
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
            ear_val = combined_ear(pts)
            t_rel = time.time() - w_start
            acc.add(ear_val, t_rel)

        # overlay info
        elapsed_in_window = time.time() - w_start
        cv2.putText(frame, f"Session: {session_id}  Label: {args.label}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Window: {window_id+1}/{args.windows}  t={elapsed_in_window:.1f}s",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        if ear_val is not None:
            cv2.putText(frame, f"EAR: {ear_val:.3f} (th={EAR_CLOSED_THRESHOLD:.2f})",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        else:
            cv2.putText(frame, "Face not detected", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.imshow("Collect Data - Drowsiness", frame)

        # cek window selesai
        if elapsed_in_window >= WINDOW_SECONDS:
            feats = acc.finalize(WINDOW_SECONDS)
            if feats is not None:
                window_id += 1
                row = {
                    "session_id": session_id,
                    "window_id": window_id,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    **feats,
                    "label": args.label
                }
                rows.append(row)
                print(f"[OK] Window {window_id} saved: {feats}")

            w_start = time.time()  # reset start window berikutnya

            if window_id >= args.windows:
                break

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # append ke CSV (kalau sudah ada)
    df_new = pd.DataFrame(rows)
    if df_new.empty:
        print("Tidak ada data tersimpan. Pastikan wajah terdeteksi.")
        return

    if os.path.exists(args.out):
        df_old = pd.read_csv(args.out)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(args.out, index=False)
    print(f"\nSelesai. Total baris baru: {len(df_new)}")
    print(f"CSV tersimpan di: {args.out}")

if __name__ == "__main__":
    main()
