import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    results = model(frame)

    mask = None

    for r in results:
        if r.masks is not None:
            mask = r.masks.data[0].cpu().numpy()

    if mask is not None:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        mask = (mask > 0.5).astype(np.uint8) * 255

        # Fond noir + sujet conservé
        foreground = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("IA Background Removal", foreground)
    else:
        cv2.imshow("IA Background Removal", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()