import numpy as np
import cv2
from funcionesV1 import *
from gestor_vehiculos import *
from vehiculos import *


def _calcular_centroide_bbox(bbox):
    """Calcula el centroide (x_c, y_c) de un bounding box (x, y, w, h)."""
    x, y, w, h = bbox
    return np.array([x + w / 2, y + h / 2])


def fusionar_detecciones_cercanas(detecciones, umbral_distancia):
    """
    Recorre una lista de bboxes y fusiona las que estén demasiado cerca.
    Se queda con la BBox más grande del grupo fusionado.
    """
    detecciones_limpias = []
    # Usamos un set para guardar los índices de las bboxes que ya hemos "usado"
    indices_usados = set()

    for i in range(len(detecciones)):
        if i in indices_usados:
            continue # Esta bbox ya fue fusionada con una anterior

        bbox_actual = detecciones[i]
        area_actual = bbox_actual[2] * bbox_actual[3]
        centroide_actual = _calcular_centroide_bbox(bbox_actual)
        
        # Esta es la BBox que guardaremos (la más grande del grupo)
        bbox_a_mantener = bbox_actual
        
        indices_usados.add(i) # Marcarla como usada

        # Ahora, comparamos esta bbox con todas las demás
        for j in range(i + 1, len(detecciones)):
            if j in indices_usados:
                continue

            bbox_comparar = detecciones[j]
            centroide_comparar = _calcular_centroide_bbox(bbox_comparar)
            
            # Calculamos la distancia euclidiana
            dist = np.linalg.norm(centroide_actual - centroide_comparar)

            if dist < umbral_distancia:
                # ¡Están demasiado cerca! Las consideramos parte del mismo objeto.
                indices_usados.add(j) # Marcamos la otra bbox como usada
                
                # Comprobamos si la nueva es más grande
                area_comparar = bbox_comparar[2] * bbox_comparar[3]
                if area_comparar > area_actual:
                    # Si es más grande, actualizamos la que vamos a guardar
                    bbox_a_mantener = bbox_comparar
                    area_actual = area_comparar
        
        # Al final del bucle 'j', añadimos la bbox más grande del grupo
        detecciones_limpias.append(bbox_a_mantener)

    return detecciones_limpias


def detectar_cochesV2(ruta_video, ruta_fondo, 
                       escala=0.5, 
                       roi_base=None,
                       umbral_sensibilidad=30, 
                       umbral_fusion_base=40,
                       min_area_base=250, 
                       kernel_size_base=7, 
                       umbral_dist_base=50, 
                       max_frames_perdido=20,
                       frames_para_confirmar=8,
                       metodo_fondo='estatico',
                       frames_calentamiento=100,
                       orientacion_via='vertical',
                       factor_perspectiva_max=3.0,
                       pixeles_por_metro = 37.1,
                       
                       filtro_sentido=None,
                       mostrar_texto_velocidad=False,
                       mostrar_texto_sentido=False,
                       mostrar_id=True,
                       mostrar_roi=True,
                       colorear_por=None,
                       vel_min_color=0.4,
                       vel_max_color=10,
                       
                       mostrar_contador_activos=True,
                       mostrar_contador_historico=True,
                       mostrar_contador_subiendo=True,
                       mostrar_contador_bajando=True,
                       mostrar_contador_motos=True,
                       mostrar_contador_coches=True,
                       mostrar_contador_camiones=True,
                       mostrar_tipo_coche=True,
                       
                       area_moto_max_base=5000,
                       area_coche_max_base=25000,):
    
    # --- Inicialización ---
    cap = leer_video(ruta_video)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # Ancho original
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Alto original
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: # Si el vídeo no tiene metadatos de FPS
        print("ADVERTENCIA: No se pudo leer el FPS del vídeo. Usando 30 por defecto.")
        fps = 30.0 # Usamos un valor estándar

    # Redimensionamos según la escala
    new_size = (int(original_width * escala), int(original_height * escala))

    if metodo_fondo == 'estatico':
        fondo_redimensionado = cv2.resize(cv2.imread(ruta_fondo), new_size).astype(np.uint8)
        print("Usando método de fondo ESTÁTICO.")

    elif metodo_fondo == 'dinamico':
        # Método nuevo: creamos un sustractor MOG2
        # history: nº de frames que usa para "aprender" el fondo.
        # varThreshold: sensibilidad (como 'umbral_sensibilidad')
        sustractor_fondo = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        print("Usando método de fondo DINÁMICO (MOG2).")
    

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
    kernel_size_val = int(np.ceil(kernel_size_base * escala)) // 2 * 2 + 1 # Ajusta el kernel (1D) a la escala y fuerza que sea impar (Truco matemático jugando con la división entera)
    kernel_escalado = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size_val, kernel_size_val)) # Crea la matriz del kernel con el tamaño escalado
    umbral_dist_escalado = umbral_dist_base * escala # Ajusta la distancia (1D) de seguimiento a la escala
    umbral_fusion_escalado = umbral_fusion_base * escala # Ajusta la distancia (1D) de la escala

    area_moto_max_escalada = area_moto_max_base * (escala**2)
    area_coche_max_escalada = area_coche_max_base * (escala**2)

    # --- Escalado de Fuentes y Posiciones de Texto ---
    # Coordenadas base (para escala 1.0)
    pos_x_base = 130
    pos_y_base = 55
    salto_y_base = 80
    pos_x_col2 = new_size[0] // 2 # Mitad de la pantalla
    
    # Fuentes base
    font_size_grande_base = 2.0
    font_size_peque_base = 0.8
    grosor_grande_base = 4
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

    # --- Lógica de control orientación ---
    if orientacion_via == 'vertical':
        label_sentido_1 = 'Subiendo'
        label_sentido_2 = 'Bajando'
        tag_sentido_1 = 'SUBE'
        tag_sentido_2 = 'BAJA'
    else: # Asumimos 'horizontal'
        label_sentido_1 = 'Izquierda'
        label_sentido_2 = 'Derecha'
        tag_sentido_1 = 'IZQUIERDA'
        tag_sentido_2 = 'DERECHA'

    frame_num = 0 # Inicializa el contador de frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1 # Incrementa el contador de fotogramas
        frame = cv2.resize(frame, new_size)

        # --- Detección dinámica (atascos) ---
        if metodo_fondo == 'dinamico' and frame_num < frames_calentamiento:
            # Si usamos MOG2 y estamos en el periodo de calentamiento...
            # 1. Alimentamos al sustractor para que aprenda
            sustractor_fondo.apply(frame)
            # 2. Mensaje de que está cargando
            print('Obteniendo el fondo dinámicamente, calentando...') 
            if cv2.waitKey(1) & 0xFF == 27: break
            # 3. Saltamos el resto del bucle (no detectar, no trackear)
            continue
        
        # --- Detección  ---
        if metodo_fondo == 'estatico':
            # Método 1: Sustracción de fondo estática
            diff = cv2.absdiff(frame, fondo_redimensionado)
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, fgmask = cv2.threshold(diff, umbral_sensibilidad, 255, cv2.THRESH_BINARY)
        
        elif metodo_fondo == 'dinamico':
            # Método 2: Sustracción dinámica (MOG2)
            # (Se ejecuta solo si frame_num >= frames_calentamiento)
            fgmask_con_sombras = sustractor_fondo.apply(frame)
            # Eliminamos las sombras (pixeles grises, valor 127) (los pixeles mayores que 250 se pasan a 255 y el resto a 0)
            _, fgmask = cv2.threshold(fgmask_con_sombras, 250, 255, cv2.THRESH_BINARY)
        
        fgmask = cv2.bitwise_and(fgmask, mask_roi) # Aplica la máscara ROI (pone a negro todo lo que esté fuera de la región)
        
        # Operaciones morfológicas para limpiar la máscara
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_escalado) # Rellena agujeros blancos dentro de los blobs
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_escalado) # Elimina pequeños puntos blancos (ruido)

        contornos, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Encuentra todos los contornos (blobs blancos) en la máscara

        # 1. Creamos la lista "sucia" de detecciones
        detecciones_sucias = []
        for c in contornos:
            if cv2.contourArea(c) < min_area_escalada:
                continue
            x, y, w, h = cv2.boundingRect(c)
            detecciones_sucias.append((x, y, w, h))
        
        # 2. Aplicamos la FUSIÓN para limpiar la lista
        detecciones_limpias = fusionar_detecciones_cercanas(detecciones_sucias, umbral_fusion_escalado)

        # 3. Actualizamos el gestor solo con las detecciones limpias
        gestor.actualizar(detecciones_limpias, frame, frame_num)
        
        # --- Contadores en tiempo real ---
        contador_actual = 0
        contador_sentido1_rt = 0
        contador_sentido2_rt = 0

        # --- Contadores de tipo ---
        contador_motos = 0
        contador_coches = 0
        contador_camiones = 0

        # Iteramos sobre los vehículos activos
        for v in gestor.vehiculos_activos():
            
            # Solo dibujamos y contamos los coches "confirmados"
            if v.frames_activo > frames_para_confirmar:
            
                x, y, w, h = map(int, v.bbox) # Obtiene la caja (bbox) del vehículo y convierte sus coordenadas a enteros
                cx, cy = v.centroide # Obtiene el centroide (x, y) suavizado por el Filtro de Kalman
                vx, vy = v.velocidad # Obtiene la velocidad (vx, vy) estimada por el Filtro de Kalman
                
                # 1. Calcular la magnitud de la velocidad (velocidad en p/f)
                velocidad_mag_bruta = np.linalg.norm(v.velocidad)
                factor_correccion = 1.0
            
                # Solo aplicamos corrección si la ROI está definida Y la vía es vertical
                if roi_base is not None and orientacion_via == 'vertical':
                    # Coordenadas Y de la ROI escalada
                    y1_roi = roi_escalada[0]
                    y2_roi = roi_escalada[1]
                    
                    # Posición Y actual del coche
                    y_coche = v.centroide[1] # v.centroide es (x,y)
                    
                    # Normalizamos la posición Y (0.0 = lejos/arriba, 1.0 = cerca/abajo)
                    # (Añadimos 1e-6 para evitar dividir por cero si la ROI es plana)
                    y_relativa = (y_coche - y1_roi) / (y2_roi - y1_roi + 1e-6)
                    y_relativa = np.clip(y_relativa, 0.0, 1.0) # Nos aseguramos que esté entre 0 y 1
                    
                    # Interpolación lineal:
                    # Si y_relativa=0 (lejos), factor = factor_perspectiva_max
                    # Si y_relativa=1 (cerca), factor = 1.0
                    factor_correccion = factor_perspectiva_max - (y_relativa * (factor_perspectiva_max - 1.0))

                # 3. Calcular la velocidad final corregida
                velocidad_corregida = velocidad_mag_bruta * factor_correccion

                # 2. Si el sentido del coche aún no está definido
                if not v.sentido and v.frames_activo > 13: # Solo fijamos la velocidad si el coche ya lleva >12 frames
                    # Comprobamos si la velocidad en Y es lo bastante fuerte para "fijarlo"
                    # Umbral +-0.5
                    if orientacion_via == 'vertical':
                        if vy < -0.5: v.sentido = tag_sentido_1 # SUBE
                        elif vy > 0.5: v.sentido = tag_sentido_2 # BAJA
                    else: # 'horizontal'
                        if vx < -0.5: v.sentido = tag_sentido_1 # IZQUIERDA
                        elif vx > 0.5: v.sentido = tag_sentido_2 # DERECHA
                
                # 3. Filtro de Sentido
                if filtro_sentido is not None and v.sentido != filtro_sentido:
                    continue
                
                # --- Si pasa todos los filtros, lo contamos y dibujamos ---
                contador_actual += 1

                # --- Lógica de Clasificación por Tipo ---
                # Si el tipo aún no está definido...
                if v.tipo == 'Indefinido':
                    x, y, w, h = v.bbox # Extraemos los valores de la caja
                    area_bbox = w * h # Calculamos el área de su BBox (w * h)
                    
                    # 1. Calcular Aspect Ratio
                    # (Añadimos 'epsilon' para evitar dividir por cero si h=0)
                    epsilon = 1e-6 
                    aspect_ratio = w / (h + epsilon) # si por error h da 0, sumarle epsilon hace que el programa no falle.
                    
                    # 2. Calcular Extent (Necesitamos el contorno original)
                    # --- Lógica de decisión ---
                    if area_bbox < area_moto_max_escalada and aspect_ratio < 0.8:
                        v.tipo = 'Moto'
                    elif area_bbox > area_coche_max_escalada and aspect_ratio > 1: # Ej: Camión
                        v.tipo = 'Camion'
                    else:
                        v.tipo = 'Coche'

                # Contamos por tipo
                if v.tipo == 'Moto': contador_motos += 1
                elif v.tipo == 'Coche': contador_coches += 1
                elif v.tipo == 'Camion': contador_camiones += 1

                # Contamos por sentido
                if v.sentido == tag_sentido_1:
                    contador_sentido1_rt += 1
                elif v.sentido == tag_sentido_2:
                    contador_sentido2_rt += 1
                
                # --- Dibujar en pantalla ---
                color_caja = (0, 255, 0) # Verde por defecto
                # -- Lógica de color de la caja --
                if colorear_por == 'sentido':
                    if v.sentido == tag_sentido_1: color_caja = (0, 255, 0) # Verde
                    elif v.sentido == tag_sentido_2: color_caja = (0, 0, 255) # Rojo
                    else: color_caja = (255, 0, 0) # Azul (detenido/indefinido)
            
                elif colorear_por == 'tipo':
                    if v.tipo == 'Moto': color_caja = (255, 0, 255) # Magenta
                    elif v.tipo == 'Camion': color_caja = (255, 255, 0) # Cyan
                    else: color_caja = (0, 255, 0) # Coche (Verde)
            
                elif colorear_por == 'velocidad':
                    # Normalizamos la velocidad (0.0 = lento, 1.0 = rápido)
                    vel_norm = (velocidad_corregida - vel_min_color) / (vel_max_color - vel_min_color)
                    vel_norm = np.clip(vel_norm, 0.0, 1.0)
                    
                    # Creamos un gradiente simple Azul -> Rojo
                    # (OpenCV es BGR, no RGB)
                    R = int(vel_norm * 255)
                    G = 0
                    B = int((1 - vel_norm) * 255)
                    color_caja = (B, G, R)

                # --- Dibujamos ---
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_caja, grosor_grande)
                
                # Mostrar sentido
                if mostrar_texto_sentido:
                    color_sentido = (255, 0, 0)
                    texto_sentido = '(...)'
                    if v.sentido == tag_sentido_1:
                        color_sentido = (0, 255, 0) # Verde
                        texto_sentido = tag_sentido_1
                    elif v.sentido == tag_sentido_2:
                        color_sentido = (0, 0, 255) # Rojo
                        texto_sentido = tag_sentido_2
                
                    y_offset = y + h + int(15 * escala)
                    if mostrar_texto_velocidad:
                        y_offset += int(15 * escala)

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
                    # --- ¡NUEVO CÁLCULO m/s! ---
                    
                    # 1. (unidades / frame) * (frames / segundo) = (unidades / segundo)
                    #    (velocidad_corregida ya es p/f en la zona 1.0)
                    velocidad_ups = velocidad_corregida * fps
                    
                    # 2. (unidades / segundo) / (unidades / metro) = (metros / segundo)
                    velocidad_ms = velocidad_ups / pixeles_por_metro
                    
                    # (Opcional: Si prefieres km/h, descomenta la siguiente línea)
                    velocidad_kmh = velocidad_ms * 3.6
                    
                    # Escribimos el resultado (ej. "8.2 m/s")
                    #cv2.putText(frame, f"{velocidad_ms:.1f} m/s", (x, y_offset), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, font_peque, (0, 255, 255), grosor_peque)
                    
                    # (Si usaste km/h, cambia la línea de arriba por esta)
                    cv2.putText(frame, f"{velocidad_kmh:.1f} km/h", (x, y_offset), 
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
        if mostrar_contador_subiendo: # (El flag se sigue llamando así, pero la etiqueta es dinámica)
            cv2.putText(frame, f"{label_sentido_1}: {contador_sentido1_rt}", (pos_x_col2, pos_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_grande, (0, 255, 0), grosor_grande)

        # --- Fila 2 ---
        # Columna 1 (Histórico)
        if mostrar_contador_historico:
            cv2.putText(frame, f"Total Historico: {contador_historico}", (pos_x, pos_y + salto_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_grande, (0, 255, 255), grosor_grande)
        
        # Columna 2 (Bajando)
        if mostrar_contador_bajando: # (Idem)
            cv2.putText(frame, f"{label_sentido_2}: {contador_sentido2_rt}", (pos_x_col2, pos_y + salto_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_grande, (0, 0, 255), grosor_grande)

        # -- Fila 3 ---
        # Columna 1 (Motos)
        if mostrar_contador_motos:
            cv2.putText(frame, f"Motos: {contador_motos}", (pos_x, pos_y + 2*salto_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_grande, (255, 0, 255), grosor_grande)
            
        # Columna 2 (Coches)
        if mostrar_contador_coches:
            cv2.putText(frame, f"Coches: {contador_coches}", (int(new_size[0] * 1/3), pos_y + 2*salto_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_grande, (255, 0, 255), grosor_grande)
            
        # Columna 3 (Camiones)
        if mostrar_contador_camiones:
            # (Lo pongo en la siguiente línea para que no se solape)
            cv2.putText(frame, f"Camiones: {contador_camiones}", (int(new_size[0] * 2/3), pos_y + 2*salto_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_grande, (255, 0, 255), grosor_grande)
            
        # Dibujamos la ROI en el frame original para debug
        if roi_escalada and mostrar_roi:
            cv2.rectangle(frame, (roi_escalada[2], roi_escalada[0]), (roi_escalada[3], roi_escalada[1]), (255, 0, 0), 2)

        cv2.imshow("Máscara", fgmask)
        cv2.imshow("Video Original", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC, ajustar waitKey para velocidad reproducción
            break

    cap.release()
    cv2.destroyAllWindows()