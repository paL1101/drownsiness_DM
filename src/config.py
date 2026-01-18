# src/config.py

# Windowing
WINDOW_SECONDS = 10.0
FPS_ASSUMPTION = 30.0  # untuk estimasi jika webcam tidak stabil

# EAR threshold (akan berbeda per orang; ini nilai awal yang cukup umum)
EAR_CLOSED_THRESHOLD = 0.20

# Minimal durasi mata tertutup (untuk menghitung kedipan), dalam detik
MIN_BLINK_CLOSED_SEC = 0.08
MAX_BLINK_CLOSED_SEC = 0.50

# Jika mata tertutup lebih lama dari ini, biasanya indikasi "ngantuk/sleepy event"
DROWSY_CLOSURE_SEC = 1.0
