import numpy as np
import cv2
from funcionesV1 import *
from gestor_vehiculos import *
from vehiculos import *

def detectar_cochesV2(ruta_video, ruta_fondo, 
                       escala=0.5, 
                       roi_base=None,
                       umbral_sensibilidad=30, 
                       min_area_base=250, 
                       kernel_size_base=7, 
                       umbral_dist_base=50, 
                       max_frames_perdido=20,
                       frames_para_confirmar=8,
                       
                       filtro_sentido=None,
                       mostrar_texto_velocidad=False,
                       mostrar_texto_sentido=False,
                       mostrar_id=True,
                       mostrar_roi=True,
                       
                       mostrar_contador_activos=True,
                       mostrar_contador_historico=True,
                       mostrar_contador_subiendo=True,
                       mostrar_contador_bajando=True):
    
    # --- Inicialización ---
    cap = leer_video(ruta_video)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # Ancho original
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Alto original

    # Redimensionamos según la escala
    new_size = (int(original_width * escala), int(original_height * escala))
    fondo_redimensionado = cv2.resize(cv2.imread(ruta_fondo), new_size).astype(np.uint8)

    # --- Definir la ROI ---
    if roi_base:
        # Si ROI pasada como parámetro
        roi_escalada = [int(roi_base[0] * escala), int(roi_base[1] * escala), 
                        int(roi_base[2] * escala), int(roi_base[3] * escala)]
        mask_roi = np.zeros((new_size[1], new_size[0]), dtype=np.uint8)
        mask_roi[roi_escalada[0]:roi_escalada[1], roi_escalada[2]:roi_escalada[3]] = 255
    else:
        # Si no tenemos ROI como parámetro
        mask_roi = np.ones((new_size[1], new_size[0]), dtype=np.uint8) * 255 # Todo bits blancos para que el bitwise_and no haga nada
        roi_escalada = None # Para que no intente dibujarla posteriormente


    # --- Cálculo de parámetros escalados ---
    min_area_escalada = min_area_base * (escala**2) # Ajusta el área (2D) de forma cuadrática a la escala
    kernel_size_val = int(np.ceil(kernel_size_base * escala)) // 2 * 2 + 1 # Ajusta el kernel (1D) a la escala y fuerza que sea impar
    kernel_escalado = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size_val, kernel_size_val)) # Crea la matriz del kernel con el tamaño escalado
    umbral_dist_escalado = umbral_dist_base * escala # Ajusta la distancia (1D) de seguimiento a la escala

    # --- Escalado de Fuentes y Posiciones de Texto ---
    # Coordenadas base (para escala 1.0)
    pos_x_base = 130
    pos_y_base = 55
    salto_y_base = 80
    pos_x_col2 = new_size[0] // 2 # Mitad de la pantalla
    
    # Fuentes base
    font_size_grande_base = 2.0
    font_size_peque_base = 0.8
    grosor_grande_base = 5
    grosor_peque_base = 3

    # Valores escalados (con un mínimo para que no desaparezcan)
    pos_x = int(pos_x_base * escala)
    pos_y = int(pos_y_base * escala)
    salto_y = int(salto_y_base * escala)

    font_grande = max(0.4, font_size_grande_base * escala)
    font_peque = max(0.4, font_size_peque_base * escala)
    grosor_grande = max(1, int(grosor_grande_base * escala))
    grosor_peque = max(1, int(grosor_peque_base * escala))

    # --- Gestor con Filtro de Kalman ---
    gestor = GestorVehiculos(
        umbral_distancia=umbral_dist_escalado, 
        max_frames_perdido=max_frames_perdido
    )

    frame_num = 0 # Inicializa el contador de frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1 # Incrementa el contador de fotogramas
        frame = cv2.resize(frame, new_size)

        # --- Detección ---
        diff = cv2.absdiff(frame, fondo_redimensionado) # Resta el fondo estático al frame actual
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # Convierte la imagen de diferencia a escala de grises
        _, fgmask = cv2.threshold(diff, umbral_sensibilidad, 255, cv2.THRESH_BINARY) # Binariza la imagen
        fgmask = cv2.bitwise_and(fgmask, mask_roi) # Aplica la máscara ROI (pone a negro todo lo que esté fuera de la región)
        
        # Operaciones morfológicas para limpiar la máscara
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_escalado) # Rellena agujeros blancos dentro de los blobs
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_escalado) # Elimina pequeños puntos blancos (ruido)

        contornos, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Encuentra todos los contornos (blobs blancos) en la máscara
        detecciones = [] # Inicializa una lista vacía para guardar las detecciones de este frame
        
        for c in contornos: # Recorre cada contorno (blob) encontrado
            if cv2.contourArea(c) < min_area_escalada:
                continue
            x, y, w, h = cv2.boundingRect(c) # Calcula la caja delimitadora (bounding box) del contorno
            detecciones.append((x, y, w, h)) # Añade las coordenadas de la caja a la lista de detecciones
        
        # --- Actualizar el gestor ---
        gestor.actualizar(detecciones, frame, frame_num)
        
        # --- Contadores en tiempo real ---
        contador_actual = 0
        contador_suben_rt = 0
        contador_bajan_rt = 0

        # Iteramos sobre los vehículos activos
        for v in gestor.vehiculos_activos():
            
            # Solo dibujamos y contamos los coches "confirmados"
            if v.frames_activo > frames_para_confirmar:
            
                x, y, w, h = map(int, v.bbox) # Obtiene la caja (bbox) del vehículo y convierte sus coordenadas a enteros
                cx, cy = v.centroide # Obtiene el centroide (x, y) suavizado por el Filtro de Kalman
                vx, vy = v.velocidad # Obtiene la velocidad (vx, vy) estimada por el Filtro de Kalman
                
                # 1. Calcular la magnitud de la velocidad (velocidad en p/f)
                velocidad_mag = np.linalg.norm(v.velocidad) # Calcula la magnitud del vector velocidad (Pitágoras) para tener un solo número (píxeles/frame)
                
                # 2. Si el sentido del coche aún no está definido
                if not v.sentido and v.frames_activo > 13: # Solo fijamos la velocidad si el coche ya lleva >12 frames
                    # Comprobamos si la velocidad en Y es lo bastante fuerte para "fijarlo"
                    # Umbral +-0.5
                    if vy < -0.5: 
                        v.sentido = 'SUBE' # Fijado
                    elif vy > 0.5:
                        v.sentido = 'BAJA' # Fijado
                
                # 3. Filtro de Sentido
                if filtro_sentido is not None and v.sentido != filtro_sentido:
                    continue
                
                # --- Si pasa todos los filtros, lo contamos y dibujamos ---
                contador_actual += 1

                color_sentido = (255, 0, 0) # Azul (por defecto, si aún es None)
                texto_sentido = '(...)'     # Texto por defecto
                
                if v.sentido == 'SUBE':
                    contador_suben_rt += 1
                    color_sentido = (0, 255, 0) # Verde
                    texto_sentido = 'SUBE'
                elif v.sentido == 'BAJA':
                    contador_bajan_rt += 1
                    color_sentido = (0, 0, 255) # Rojo
                    texto_sentido = 'BAJA'

                # --- Dibujar en pantalla ---
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), grosor_grande)
                
                # ID del vehículo (este está por ENCIMA de la caja, así que no afecta al offset)
                if mostrar_id:
                    cv2.putText(frame, f"ID {v.id}", (x, y - int(10 * escala)), # Escalamos también el y-10
                                cv2.FONT_HERSHEY_SIMPLEX, font_peque, (255,255,0), grosor_peque)
                
                # --- Lógica de Offset Dinámico ---
                # 1. Definimos el salto de línea base (escalado)
                salto_linea = int(24 * escala) # (ej. 15 píxeles para escala 1.0)
                
                # 2. Inicializamos el offset Y (justo debajo de la caja)
                y_offset = y + h + salto_linea
                
                # Velocidad (magnitud)
                if mostrar_texto_velocidad:
                    cv2.putText(frame, f"{velocidad_mag:.1f} p/f", (x, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_peque, (0, 255, 255), grosor_peque)
                    
                    # 3. Incrementamos el offset SOLO SI hemos dibujado
                    y_offset += salto_linea 
                
                # Sentido (ahora usa las variables 'texto_sentido' y 'color_sentido')
                if mostrar_texto_sentido:
                    # 4. Dibuja en la última posición del offset
                    # (Si velocidad no se mostró, y_offset = y+h+15)
                    # (Si velocidad SÍ se mostró, y_offset = y+h+30)
                    cv2.putText(frame, texto_sentido, (x, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_peque, color_sentido, grosor_peque)
        
        # --- Dibujar Contadores Globales ---
        # Obtenemos el histórico (solo si se va a mostrar)
        if mostrar_contador_historico:
            contador_historico = Vehiculo._next_id

        # --- Fila 1 ---
        # Columna 1 (Activos)
        if mostrar_contador_activos:
            cv2.putText(frame, f"Vehiculos Activos: {contador_actual}", (pos_x, pos_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_grande, (0, 255, 255), grosor_grande)
        
        # Columna 2 (Subiendo)
        if mostrar_contador_subiendo:
            cv2.putText(frame, f"Subiendo: {contador_suben_rt}", (pos_x_col2, pos_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_grande, (0, 255, 0), grosor_grande)

        # --- Fila 2 ---
        # Columna 1 (Histórico)
        if mostrar_contador_historico:
            cv2.putText(frame, f"Total Historico: {contador_historico}", (pos_x, pos_y + salto_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_grande, (0, 255, 255), grosor_grande)
        
        # Columna 2 (Bajando)
        if mostrar_contador_bajando:
            cv2.putText(frame, f"Bajando: {contador_bajan_rt}", (pos_x_col2, pos_y + salto_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_grande, (0, 0, 255), grosor_grande)

        # Dibujamos la ROI en el frame original para debug
        if roi_escalada and mostrar_roi:
            cv2.rectangle(frame, (roi_escalada[2], roi_escalada[0]), (roi_escalada[3], roi_escalada[1]), (255, 0, 0), 2)

        cv2.imshow("Máscara", fgmask)
        cv2.imshow("Video Original", frame)

        if cv2.waitKey(30) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()