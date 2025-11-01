import numpy as np
import cv2
from funcionesV1 import *
from gestor_vehiculos import *
from vehiculos import Vehiculo, Coche # Importamos Coche también

def detectar_cochesV2(ruta_video, background_img_path, 
                       escala=0.5, 
                       umbral_sensibilidad=30, 
                       min_area_base=250, 
                       kernel_size_base=7, 
                       umbral_dist_base=50, 
                       max_frames_perdido=20,
                       frames_para_confirmar=8):
                       # --- Eliminamos los parámetros de 'zona_superior_y_base' ---
    
    # --- Inicialización ---
    cap = leer_video(ruta_video)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_size = (int(original_width * escala), int(original_height * escala))
    
    fondo_redimensionado = cv2.resize(cv2.imread(background_img_path), new_size).astype(np.uint8)

    # --- Definir la ROI (Región de Interés) ---
    roi_base = [280, 965, 0, 1920] 
    roi_escalada = [int(roi_base[0] * escala), int(roi_base[1] * escala), 
                    int(roi_base[2] * escala), int(roi_base[3] * escala)]
    mask_roi = np.zeros((new_size[1], new_size[0]), dtype=np.uint8)
    mask_roi[roi_escalada[0]:roi_escalada[1], roi_escalada[2]:roi_escalada[3]] = 255


    # --- Cálculo de parámetros escalados ---
    min_area_escalada = min_area_base * (escala**2)
    kernel_size_val = int(np.ceil(kernel_size_base * escala)) // 2 * 2 + 1
    kernel_escalado = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size_val, kernel_size_val))
    umbral_dist_escalado = umbral_dist_base * escala

    # --- Gestor con Filtro de Kalman ---
    gestor = GestorVehiculos(
        umbral_distancia=umbral_dist_escalado, 
        max_frames_perdido=max_frames_perdido
    )

    frame_num = 0
    
    # --- Contadores Globales ---
    # (Ya no necesitamos los contadores de salida, se calcularán en cada frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        frame = cv2.resize(frame, new_size)

        # --- Detección (todo igual) ---
        diff = cv2.absdiff(frame, fondo_redimensionado)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, fgmask = cv2.threshold(diff, umbral_sensibilidad, 255, cv2.THRESH_BINARY)
        fgmask = cv2.bitwise_and(fgmask, mask_roi)
        
        # Dejamos el orden que te funcionaba bien
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_escalado)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_escalado)

        contornos, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detecciones = []
        for c in contornos:
            if cv2.contourArea(c) < min_area_escalada:
                continue
            x, y, w, h = cv2.boundingRect(c)
            detecciones.append((x, y, w, h))
        
        # --- Actualizar el gestor (Ahora usa Kalman) ---
        gestor.actualizar(detecciones, frame, frame_num)


        # --- Dibujar resultados (CON LÓGICA DE VELOCIDAD) ---
        
        contador_actual = 0
        
        # --- NUEVO: Contadores de sentido en tiempo real ---
        contador_suben_rt = 0
        contador_bajan_rt = 0

        # Iteramos sobre los vehículos activos
        for v in gestor.vehiculos_activos():
            
            # Solo dibujamos y contamos los coches "confirmados"
            if v.frames_activo > frames_para_confirmar:
                
                contador_actual += 1
            
                x, y, w, h = map(int, v.bbox)
                # Obtenemos el centroide desde el estado del filtro
                cx, cy = v.centroide 
                
                # --- ¡AQUÍ ESTÁ LA MAGIA! ---
                # Obtenemos la velocidad (vx, vy) desde el estado del filtro
                vx, vy = v.velocidad
                
                # 1. Calcular la magnitud de la velocidad (velocidad en p/f)
                velocidad_mag = np.linalg.norm(v.velocidad)
                
                # 2. Determinar el sentido
                sentido = None
                # Usamos un umbral (ej. 0.5) para ignorar velocidades en Y muy pequeñas
                if vy < -0.5:
                    sentido = 'SUBE'
                    contador_suben_rt += 1
                elif vy > 0.5:
                    sentido = 'BAJA'
                    contador_bajan_rt += 1

                # --- Dibujar en pantalla ---
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # ID del vehículo
                cv2.putText(frame, f"ID {v.id}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                
                # Velocidad (magnitud)
                cv2.putText(frame, f"{velocidad_mag:.1f} p/f", (x, y + h + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # Sentido
                if sentido == 'SUBE':
                    color_sentido = (0, 255, 0) # Verde
                elif sentido == 'BAJA':
                    color_sentido = (0, 0, 255) # Rojo
                else:
                    color_sentido = (255, 0, 0) # Azul (detenido)
                
                cv2.putText(frame, sentido, (x, y + h + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_sentido, 1)
        
        # Dibujar contadores en pantalla
        # Obtenemos el contador histórico (total de IDs únicos creados)
        contador_historico = Vehiculo._next_id
        # (Ajusta las coordenadas (20, 50) y (20, 90) si lo necesitas)
        cv2.putText(frame, f"Vehiculos Activos: {contador_actual}", (65, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Total Historico: {contador_historico}", (65, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        # --- Dibujar nuevos contadores de sentido (en tiempo real) ---
        cv2.putText(frame, f"Subiendo: {contador_suben_rt}", (410, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Bajando: {contador_bajan_rt}", (410, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Dibujamos la ROI en el frame original para debug
        cv2.rectangle(frame, (roi_escalada[2], roi_escalada[0]), (roi_escalada[3], roi_escalada[1]), (255, 0, 0), 2)

        cv2.imshow("Máscara", fgmask)
        cv2.imshow("Video Original", frame)

        if cv2.waitKey(30) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()