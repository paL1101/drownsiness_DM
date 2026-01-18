# app.py - FIXED untuk Gradio 6.0 (dengan parameter yang benar)
import os
import time
import numpy as np
import cv2
import joblib
import gradio as gr
import mediapipe as mp

# Import dari src/
from src.config import WINDOW_SECONDS, EAR_CLOSED_THRESHOLD
from src.features import combined_ear, WindowAccumulator

FEATURES = [
    "ear_mean", "ear_min", "ear_std",
    "perclos", "blink_count",
    "avg_closure_ms", "max_closure_ms"
]

RF_MODEL_PATH = os.path.join("models", "rf_model.pkl")
TASK_MODEL_PATH = os.path.join("models", "mediapipe", "face_landmarker.task")

print(f"Model path: {RF_MODEL_PATH}")
print(f"Task path: {TASK_MODEL_PATH}")
print(f"Model exists: {os.path.exists(RF_MODEL_PATH)}")
print(f"Task exists: {os.path.exists(TASK_MODEL_PATH)}")

# --- MediaPipe Setup ---
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

def create_landmarker():
    """Create MediaPipe FaceLandmarker"""
    try:
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
        landmarker = FaceLandmarker.create_from_options(options)
        print("✓ FaceLandmarker initialized")
        return landmarker
    except Exception as e:
        print(f"✗ Error creating landmarker: {e}")
        return None

def draw_label(img_bgr, lines, x=12, y=30):
    """Draw text with high contrast"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2

    # Calculate box size
    sizes = [cv2.getTextSize(t, font, scale, thickness)[0] for t in lines]
    w = max(s[0] for s in sizes) + 20
    h = (len(lines) * 28) + 16

    # Draw black background box
    cv2.rectangle(img_bgr, (x-10, y-24), (x-10+w, y-24+h), (0, 0, 0), -1)

    # Draw text with white outline
    yy = y
    for text in lines:
        # White outline
        cv2.putText(img_bgr, text, (x, yy), font, scale, (255, 255, 255), thickness+2, cv2.LINE_AA)
        # Black text on top
        cv2.putText(img_bgr, text, (x, yy), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
        yy += 28

    return img_bgr

def make_demo():
    """Create inference function"""
    
    # Load model
    try:
        rf = joblib.load(RF_MODEL_PATH)
        print("✓ Model loaded")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

    # Create landmarker
    landmarker = create_landmarker()
    if landmarker is None:
        print("✗ Cannot create landmarker")
        return None

    # Initialize state
    acc = WindowAccumulator(EAR_CLOSED_THRESHOLD)
    window_start = time.time()
    start_ts = time.time()
    
    last_pred = None
    last_prob = 0.0
    last_ear = None

    def infer(frame_rgb):
        """Process frame and return output"""
        nonlocal window_start, start_ts, last_pred, last_prob, last_ear, acc

        if frame_rgb is None:
            print("✗ No frame received")
            return None, "No frame"

        try:
            # Get frame dimensions
            h, w = frame_rgb.shape[:2]
            # print(f"Frame received: {w}x{h}")  # Commented out to reduce console spam

            # Convert to MediaPipe format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Detect landmarks
            timestamp_ms = int((time.time() - start_ts) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            ear_val = None
            if result.face_landmarks and len(result.face_landmarks) > 0:
                lm = result.face_landmarks[0]
                pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
                ear_val = float(combined_ear(pts))
                acc.add(ear_val, time.time() - window_start)

            # Check if window complete
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
                    print(f"Prediction: {last_pred}, Prob: {last_prob:.3f}")
                
                window_start = time.time()

            last_ear = ear_val

            # Prepare output
            img_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Determine status
            status = "—"
            if last_pred is not None:
                status = "NGANTUK" if last_pred == 1 else "TIDAK NGANTUK"

            # Prepare info lines
            lines = [
                f"Status: {status}   P(ngantuk): {last_prob:.2f}",
                f"EAR: {last_ear:.3f}" if last_ear is not None else "EAR: (detecting...)",
                f"Window: {min(elapsed, WINDOW_SECONDS):.1f}/{WINDOW_SECONDS:.0f}s"
            ]

            # Draw overlay
            img_bgr = draw_label(img_bgr, lines)

            # Convert back to RGB for Gradio
            img_out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            ear_display = f"{last_ear:.3f}" if last_ear else "0.000"
            output_text = f"{status} | P={last_prob:.2f} | EAR={ear_display}"
            
            return img_out, output_text

        except Exception as e:
            print(f"✗ Error in infer: {e}")
            import traceback
            traceback.print_exc()
            return frame_rgb, f"Error: {str(e)}"

    return infer

# --- Gradio Interface ---
print("\n🚀 Starting Gradio app...")
print(f"Current directory: {os.getcwd()}")

infer_fn = make_demo()

if infer_fn is None:
    print("✗ Cannot start app - infer function is None")
else:
    # Create blocks without theme parameter
    demo = gr.Blocks(title="Drowsiness Detection (RF + FaceLandmarker)")
    
    with demo:
        gr.Markdown("# 😴 Drowsiness Detection")
        gr.Markdown("### Real-time Detection with Random Forest + MediaPipe")
        gr.Markdown("Gunakan kamera browser untuk pembuktian realtime.")

        with gr.Row():
            with gr.Column():
                # FIX: Remove mirror_webcam parameter - it doesn't exist in Gradio 6.0
                webcam_input = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    label="📹 Webcam Input"
                )

            with gr.Column():
                image_output = gr.Image(
                    label="📊 Output (dengan overlay)",
                    type="numpy"
                )
                text_output = gr.Textbox(
                    label="📈 Status & Result",
                    interactive=False,
                    lines=1
                )

        # Start streaming
        webcam_input.stream(
            infer_fn,
            inputs=webcam_input,
            outputs=[image_output, text_output],
            time_limit=3600  # 1 hour limit
        )

        # Add info section
        gr.Markdown("---")
        gr.Markdown("## 📋 Penjelasan:")
        gr.Markdown("""
        - **Status**: NGANTUK atau TIDAK NGANTUK
        - **P(ngantuk)**: Probability kengantukan (0.0-1.0)
        - **EAR**: Eye Aspect Ratio (Eye opening level)
        - **Window**: Progress menuju prediksi berikutnya (10 detik)
        
        **Features:**
        - Real-time webcam streaming
        - 468 facial landmarks detection (MediaPipe)
        - Random Forest classification
        - High-contrast overlay display
        """)

    # Launch app
    print("\n✅ Launching Gradio interface...")
    print("Open your browser at: http://127.0.0.1:7860")
    print("\nPress Ctrl+C to stop the server")
    
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )