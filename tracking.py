import cv2
from ultralytics import YOLO

# Charger le modèle IA pré-entraîné (détection + tracking)
model = YOLO("yolov8n.pt")  # petit modèle rapide

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Webcam non détectée")
    exit()

print("Tracking IA lancé... (Appuie sur Q pour quitter)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Tracking automatique avec IA
    results = model.track(frame, persist=True)

    # Affichage avec les boîtes + ID de tracking
    annotated_frame = results[0].plot()

    cv2.imshow("Tracking IA - YOLOv8", annotated_frame)

    # Quitter avec Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()