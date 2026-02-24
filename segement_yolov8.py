import cv2
import numpy as np
from ultralytics import YOLO

# Charger le modèle segmentation (IMPORTANT : -seg)
model = YOLO("yolov8n-seg.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur webcam")
    exit()

print("Segmentation contour IA lancée...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        if r.masks is not None:

            # On récupère chaque masque
            for mask in r.masks.data:

                # Convertir en numpy
                mask = mask.cpu().numpy()

                # Redimensionner au format image
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                # Binariser
                mask = (mask > 0.5).astype(np.uint8) * 255

                # Trouver le contour exact
                contours, _ = cv2.findContours(
                    mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                # Dessiner le contour précis
                cv2.drawContours(
                    frame,
                    contours,
                    -1,
                    (0, 255, 0),
                    3
                )

    cv2.imshow("Segmentation Contour Precise", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()