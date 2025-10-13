from utilidades_basicas import *


def contar_coches(video, fondo, ancho, alto, linea_y, min_area, min_dist):
    """
    Cuenta los coches (blobs) del vídeo <vídeo> que cruzan una línea <linea_y> en la carretera utilizando
    una distancia minima <min_dist> para considerar un blob el mismo coche.
    """

    cap = leer_video(video)
    fondo = cv2.resize(cv2.imread(fondo), (ancho, alto)).astype(np.uint8)

    # Definimos la ROI según el vídeo
    x1, y1, x2, y2 = 0, 125, 800, 450

    # Zona a ignorar (timestamp u overlay de texto)
    x_txt, y_txt, w_txt, h_txt = 675, 400, 30, 20  # coordenadas del recuadro de los números

    # Variables de seguimiento
    vehiculos_contados = []  # [(cx, cy, f)] ya contados
    contador = 0
    tiempo_memoria = 15 # en frames (entre 15 y 20 para este video)
    frame_actual = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del vídeo")
            break
        frame_actual += 1

        frame = cv2.resize(frame, (ancho, alto))

        # Eliminar el timestamp (píxeles cambiantes de la fecha/hora) (explicado en detectar_coches)
        cv2.rectangle(frame, (x_txt, y_txt), (x_txt + w_txt, y_txt + h_txt), (0, 0, 0), -1)
        cv2.rectangle(fondo, (x_txt, y_txt), (x_txt + w_txt, y_txt + h_txt), (0, 0, 0), -1)

        diferencia = cv2.absdiff(frame, fondo)

        # Convertimos a escala de grises (solo necesitamos intensidad, no color)
        gris = cv2.cvtColor(diferencia, cv2.COLOR_BGR2GRAY)

        # Umbralizamos para quedarnos con zonas de movimiento (explicado en detectar_coches)
        _, umbral = cv2.threshold(gris, 33, 255, cv2.THRESH_BINARY)

        # Aplicamos operaciones morfológicas (explicado en detectar_coches)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        umbral = cv2.morphologyEx(umbral, cv2.MORPH_CLOSE, kernel)  # cierra huecos
        umbral = cv2.morphologyEx(umbral, cv2.MORPH_OPEN, kernel)   # quita ruido

        #Aplicamos la máscara de la ROI (explicado en detectar_coches)
        mask = np.zeros_like(umbral)
        mask[y1:y2, x1:x2] = 255
        umbral_roi = cv2.bitwise_and(umbral, mask)

        # Buscamos contornos (posibles coches) (explicado en detectar_coches)
        contornos, _ = cv2.findContours(umbral_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vehiculos_actuales = []

        # Detección de blobs
        for cont in contornos:
            if cv2.contourArea(cont) > min_area:
                x, y, w, h = cv2.boundingRect(cont)
                cx, cy = int(x + w / 2), int(y + h / 2) # coordenadas del centro del blob
                vehiculos_actuales.append((cx, cy)) # añadimos a la lista la coordenada de su centro
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # rectángulo verde
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1) # círculo (punto) rojo en el centro del blob
        
        # Comprobación de cruce de línea
        for (cx, cy) in vehiculos_actuales:
            if abs(cy - linea_y) < 4:  # margen de cruce de linea (si es menor que el margen se intuye que ya pasó)
                ya_contado = False
                for (px, py, f) in vehiculos_contados:
                    if abs(cx - px) < min_dist and abs(cy - py) < min_dist and (frame_actual - f) < tiempo_memoria: # Si dos centros están muy cerca signfica que son el mismo coche.
                        ya_contado = True
                        break

                if not ya_contado:
                    contador += 1
                    vehiculos_contados.append((cx, cy, frame_actual))
                    print(f"Vehículo #{contador} detectado (frame {frame_actual})")

        # Eliminamos coches viejos del historial
        vehiculos_contados_actualizado = []
        for (x, y, f) in vehiculos_contados:
            # Si el coche fue detectado hace menos de 'tiempo_memoria' frames, lo mantenemos
            if frame_actual - f < tiempo_memoria:
                vehiculos_contados_actualizado.append((x, y, f))
        vehiculos_contados = vehiculos_contados_actualizado

        # Dibujar información
        cv2.line(frame, (0, linea_y), (ancho, linea_y), (255, 0, 0), 2)
        cv2.putText(frame, f"Vehiculos: {contador}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Conteo Vehiculos', frame)
        cv2.imshow('Mascara Movimiento', umbral)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC para salir
            break

    print(f"\nTotal de vehículos que cruzaron la línea: {contador}")
    cap.release()
    cv2.destroyAllWindows()