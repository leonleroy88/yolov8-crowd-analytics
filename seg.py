import cv2
from ultralytics import YOLO

# Charger un modèle de segmentation pré-entraîné (léger et rapide)
model = YOLO("yolov8n-seg.pt")

print("Modele YOLO charge avec succes !")

# Ouvrir la webcam (0 = webcam par defaut)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionner pour de meilleures performances
    frame = cv2.resize(frame, (640, 480))

    # Prédiction IA (segmentation en temps réel)
    results = model(frame, stream=True)

    # Affichage des résultats (masques + bounding boxes)
    for r in results:
        annotated_frame = r.plot()

    # Affichage de la fenêtre
    cv2.imshow("Segmentation IA Temps Reel (YOLOv8)", annotated_frame)

    # Quitter avec la touche Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()