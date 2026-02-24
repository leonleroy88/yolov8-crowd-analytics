import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time

# Charger modèle
model = YOLO("yolov8n-seg.pt")

# --- CHANGE TON FICHIER ICI ---
VIDEO_PATH = "times_square.mp4"
SEUIL_ALERTE = 10  # Nombre de personnes avant alerte rouge

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Erreur : fichier vidéo non trouvé")
    exit()

# Dimensions vidéo
ret, test_frame = cap.read()
if not ret:
    exit()
VIDEO_H, VIDEO_W = test_frame.shape[:2]
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Taille dashboard
DASH_W = 400
TOTAL_W = VIDEO_W + DASH_W

# Heatmap accumulée
heatmap_acc = np.zeros((VIDEO_H, VIDEO_W), dtype=np.float32)

# Historique comptage pour graphique
historique_count = deque(maxlen=100)
historique_temps = deque(maxlen=100)
temps_debut = time.time()

print("Analyse de foule lancée...")

def dessiner_dashboard(dashboard, count, historique_count, alerte):
    dashboard[:] = (15, 15, 25)  # fond sombre

    # Titre
    cv2.putText(dashboard, "CROWD ANALYTICS", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.line(dashboard, (20, 55), (DASH_W - 20, 55), (50, 50, 80), 1)

    # Compteur principal
    couleur_count = (0, 80, 255) if alerte else (0, 255, 120)
    cv2.putText(dashboard, "PERSONNES DETECTEES", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(dashboard, str(count), (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 3.5, couleur_count, 6)

    # Alerte
    if alerte:
        cv2.rectangle(dashboard, (10, 175), (DASH_W - 10, 210), (0, 0, 180), -1)
        cv2.putText(dashboard, f"⚠ ALERTE > {SEUIL_ALERTE} personnes", (18, 198),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Séparateur
    cv2.line(dashboard, (20, 225), (DASH_W - 20, 225), (50, 50, 80), 1)

    # Graphique historique
    cv2.putText(dashboard, "HISTORIQUE (100 frames)", (20, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    if len(historique_count) > 1:
        graph_x, graph_y = 20, 265
        graph_w, graph_h = DASH_W - 40, 120
        cv2.rectangle(dashboard, (graph_x, graph_y),
                      (graph_x + graph_w, graph_y + graph_h), (30, 30, 50), -1)

        max_val = max(max(historique_count), 1)
        points = []
        for i, val in enumerate(historique_count):
            px = graph_x + int(i * graph_w / len(historique_count))
            py = graph_y + graph_h - int(val * graph_h / max_val)
            points.append((px, py))

        for i in range(1, len(points)):
            couleur_ligne = (0, 80, 255) if historique_count[i] >= SEUIL_ALERTE else (0, 255, 120)
            cv2.line(dashboard, points[i-1], points[i], couleur_ligne, 2)

        # Axe Y label
        cv2.putText(dashboard, str(max_val), (graph_x + graph_w + 5, graph_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        cv2.putText(dashboard, "0", (graph_x + graph_w + 5, graph_y + graph_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    # Séparateur
    cv2.line(dashboard, (20, 405), (DASH_W - 20, 405), (50, 50, 80), 1)

    # Stats
    cv2.putText(dashboard, "STATS SESSION", (20, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    if historique_count:
        moy = int(sum(historique_count) / len(historique_count))
        maxi = int(max(historique_count))
        cv2.putText(dashboard, f"Moyenne : {moy} pers.", (20, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(dashboard, f"Pic max : {maxi} pers.", (20, 490),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Seuil alerte info
    cv2.putText(dashboard, f"Seuil alerte : {SEUIL_ALERTE} pers.", (20, 525),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    return dashboard

while True:
    ret, frame = cap.read()
    if not ret:
        # Fin de vidéo → recommencer
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    results = model(frame, classes=[0], verbose=False)  # classes=0 = personnes uniquement

    count = 0
    heatmap_frame = np.zeros((VIDEO_H, VIDEO_W), dtype=np.float32)

    for r in results:
        # Compter via les boîtes (plus fiable)
        if r.boxes is not None:
            count = len(r.boxes)

        # Heatmap via masques
        if r.masks is not None:
            for mask in r.masks.data:
                mask_np = mask.cpu().numpy()
                mask_np = cv2.resize(mask_np, (VIDEO_W, VIDEO_H))
                heatmap_frame += mask_np

    # Accumuler heatmap
    heatmap_acc += heatmap_frame
    heatmap_acc *= 0.95  # decay pour ne pas saturer

    # Normaliser et coloriser heatmap
    heatmap_norm = cv2.normalize(heatmap_acc, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)

    # Fusionner vidéo + heatmap
    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

    # Historique
    historique_count.append(count)

    # Alerte
    alerte = count >= SEUIL_ALERTE

    # Dashboard
    dashboard = np.zeros((VIDEO_H, DASH_W, 3), dtype=np.uint8)
    dashboard = dessiner_dashboard(dashboard, count, historique_count, alerte)

    # Assembler vidéo + dashboard
    final = np.hstack([overlay, dashboard])

    cv2.imshow("Crowd Analytics - YOLOv8", final)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    if cv2.getWindowProperty("Crowd Analytics - YOLOv8", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
print("Analyse terminée.")