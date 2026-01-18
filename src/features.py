# src/features.py
import numpy as np

# Indeks landmark MediaPipe FaceMesh untuk area mata.
# Ini adalah subset yang umum digunakan untuk menghitung EAR secara stabil.
# Catatan: FaceMesh punya 468 titik; indeks di bawah adalah pasangan yang lazim dipakai.
LEFT_EYE = {
    "p1": 33,   # outer corner
    "p4": 133,  # inner corner
    "p2": 160,  # upper
    "p6": 144,  # lower
    "p3": 158,  # upper
    "p5": 153,  # lower
}
RIGHT_EYE = {
    "p1": 362,  # outer corner
    "p4": 263,  # inner corner
    "p2": 385,  # upper
    "p6": 380,  # lower
    "p3": 387,  # upper
    "p5": 373,  # lower
}

def _dist(a, b) -> float:
    return float(np.linalg.norm(a - b))

def ear_from_landmarks(landmarks_xy: np.ndarray, which: str = "left") -> float:
    """
    landmarks_xy: (468, 2) array, koordinat pixel (x,y) dari face mesh.
    which: "left" atau "right"
    """
    eye = LEFT_EYE if which == "left" else RIGHT_EYE
    p1 = landmarks_xy[eye["p1"]]
    p4 = landmarks_xy[eye["p4"]]
    p2 = landmarks_xy[eye["p2"]]
    p6 = landmarks_xy[eye["p6"]]
    p3 = landmarks_xy[eye["p3"]]
    p5 = landmarks_xy[eye["p5"]]

    # EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    denom = 2.0 * _dist(p1, p4)
    if denom <= 1e-6:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / denom

def combined_ear(landmarks_xy: np.ndarray) -> float:
    """Rata-rata EAR mata kiri dan kanan."""
    l = ear_from_landmarks(landmarks_xy, "left")
    r = ear_from_landmarks(landmarks_xy, "right")
    return (l + r) / 2.0


class WindowAccumulator:
    """
    Mengumpulkan EAR per frame dalam satu window (mis. 10 detik),
    lalu menghitung fitur tabular untuk 1 baris CSV.
    """
    def __init__(self, ear_closed_threshold: float):
        self.th = ear_closed_threshold
        self.ears = []
        self.closed_flags = []
        self.timestamps = []  # detik relatif
        self._closure_segments = []  # list of (start_t, end_t)

        self._in_closed = False
        self._closed_start_t = None

    def add(self, ear: float, t: float):
        self.ears.append(float(ear))
        is_closed = ear < self.th
        self.closed_flags.append(int(is_closed))
        self.timestamps.append(float(t))

        # deteksi segmen mata tertutup
        if is_closed and not self._in_closed:
            self._in_closed = True
            self._closed_start_t = t

        if (not is_closed) and self._in_closed:
            self._in_closed = False
            self._closure_segments.append((self._closed_start_t, t))
            self._closed_start_t = None

    def finalize(self, window_seconds: float):
        # jika window berakhir dan mata masih tertutup, tutup segmennya
        if self._in_closed and self._closed_start_t is not None and len(self.timestamps) > 0:
            self._closure_segments.append((self._closed_start_t, self.timestamps[-1]))

        ears = np.array(self.ears, dtype=np.float32)
        if len(ears) == 0:
            return None

        ear_mean = float(np.mean(ears))
        ear_min = float(np.min(ears))
        ear_std = float(np.std(ears))

        # PERCLOS: proporsi waktu tertutup dalam window
        # pendekatan sederhana: berdasarkan closed_flags per frame (cukup untuk UAS)
        perclos = float(np.mean(np.array(self.closed_flags, dtype=np.float32)))

        # Durasi segmen mata tertutup
        closure_durations = [max(0.0, end - start) for start, end in self._closure_segments]
        closure_ms = [d * 1000.0 for d in closure_durations]

        if len(closure_ms) > 0:
            avg_closure_ms = float(np.mean(closure_ms))
            max_closure_ms = float(np.max(closure_ms))
        else:
            avg_closure_ms = 0.0
            max_closure_ms = 0.0

        # Blink count: segmen tertutup dengan durasi blink wajar
        # (kedipan biasanya 80–500 ms)
        blink_count = 0
        for d in closure_durations:
            if 0.08 <= d <= 0.50:
                blink_count += 1

        # Reset untuk window berikutnya
        self.ears.clear()
        self.closed_flags.clear()
        self.timestamps.clear()
        self._closure_segments.clear()
        self._in_closed = False
        self._closed_start_t = None

        return {
            "ear_mean": ear_mean,
            "ear_min": ear_min,
            "ear_std": ear_std,
            "perclos": perclos,
            "blink_count": int(blink_count),
            "avg_closure_ms": avg_closure_ms,
            "max_closure_ms": max_closure_ms,
        }
