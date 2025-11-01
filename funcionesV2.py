import numpy as np
import cv2
from funcionesV1 import *
from gestor_vehiculos import *

def detectar_cochesV2(ruta_video, background_img_path, 
                       escala=0.5, 
                       umbral_sensibilidad=50, 
                       min_area_base=60, 
                       kernel_size_base=5, 
                       umbral_dist_base=50, 
                       max_frames_perdido=10):
    """
    Procesa un vídeo detectando y siguiendo vehículos.
    Todos los parámetros de detección y seguimiento se pasan como argumentos.
    """

    # --- Inicialización ---
    cap = leer_video(ruta_video)

    # --- Cálculo de dimensiones basado en la escala ---
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_size = (int(original_width * escala), int(original_height * escala))
    
    fondo_redimensionado = cv2.resize(cv2.imread(background_img_path), new_size).astype(np.uint8)

    # --- NUEVO: Definir la ROI (Región de Interés) ---
    # Coordenadas [y1, y2, x1, x2] basadas en el tamaño ORIGINAL (1920x1080)
    # Ignoramos el cielo (top 300px) y el timestamp (bottom 50px)
    roi_base = [200, 965, 0, 1920] 
    
    # Escalamos la ROI
    roi_escalada = [
        int(roi_base[0] * escala), # y1
        int(roi_base[1] * escala), # y2
        int(roi_base[2] * escala), # x1
        int(roi_base[3] * escala)  # x2
    ]
    # Creamos la máscara de la ROI una sola vez
    mask_roi = np.zeros((new_size[1], new_size[0]), dtype=np.uint8)
    mask_roi[roi_escalada[0]:roi_escalada[1], roi_escalada[2]:roi_escalada[3]] = 255


    # --- Cálculo de parámetros escalados (basados en los argumentos) ---
    min_area_escalada = min_area_base * (escala**2)
    kernel_size_val = int(np.ceil(kernel_size_base * escala)) // 2 * 2 + 1
    kernel_escalado = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size_val, kernel_size_val))
    umbral_dist_escalado = umbral_dist_base * escala

    gestor = GestorVehiculos(
        umbral_distancia=umbral_dist_escalado, 
        max_frames_perdido=max_frames_perdido
    )

    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        frame = cv2.resize(frame, new_size)

        diff = cv2.absdiff(frame, fondo_redimensionado)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        _, fgmask = cv2.threshold(diff, umbral_sensibilidad, 255, cv2.THRESH_BINARY)

        # --- NUEVO: Aplicamos la máscara de la ROI ---
        # Esto elimina el timestamp y el ruido de los árboles/cielo
        fgmask = cv2.bitwise_and(fgmask, mask_roi)

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_escalado)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_escalado)

        contornos, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detecciones = []

        for c in contornos:
            if cv2.contourArea(c) < min_area_escalada:
                continue
            x, y, w, h = cv2.boundingRect(c)
            detecciones.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        
        gestor.actualizar(detecciones, frame, frame_num)

        # --- Dibujar resultados ---
        for v in gestor.vehiculos_activos():
            x, y, w, h = map(int, v.bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {v.id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            
            trazado_min = max(1, len(v.historial) - 19)
            for i in range(trazado_min, len(v.historial)):
                cv2.line(frame,
                         (int(v.historial[i-1][0]), int(v.historial[i-1][1])),
                         (int(v.historial[i][0]), int(v.historial[i][1])),
                         (0, 255, 255), 1)
        
        # Dibujamos la ROI en el frame original para debug
        cv2.rectangle(frame, (roi_escalada[2], roi_escalada[0]), (roi_escalada[3], roi_escalada[1]), (255, 0, 0), 2)

        cv2.imshow("Máscara", fgmask)
        cv2.imshow("Video Original", frame)

        if cv2.waitKey(30) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()