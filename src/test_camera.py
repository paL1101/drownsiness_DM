import cv2

def try_open(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # Windows: lebih stabil
    if not cap.isOpened():
        return False
    ok, frame = cap.read()
    cap.release()
    return ok and frame is not None

for i in range(6):
    print(i, "OK" if try_open(i) else "FAIL")
