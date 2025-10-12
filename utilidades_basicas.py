import cv2
import numpy as np
import matplotlib.pyplot as plt


def leer_video(video):
    """
    Lee un vídeo desde la ruta <video> y devuelve el objeto de captura (cv2.VideoCapture).
    Muestra un mensaje de error y finaliza el programa si no se puede abrir.
    """
    cap = cv2.VideoCapture(video)

    # Mensaje por si no podemos acceder al archivo
    if not cap.isOpened():
        print("Error!")
        exit()
    
    return cap


def visualizar_video(video, ancho, alto):
    """Muestra un vídeo <video> redimensionado (<ancho>, <alto>) hasta que termine o se pulse ESC"""
    cap = leer_video(video)

    # Visualizar el vídeo
    while(True):
        
        ret, frame = cap.read()
        
        # Comprobamos que se esté visualizando correctamente
        if not ret:
            print("Fin del vídeo")
            break

        # Ajustamos para que el vídeo ocupe menos
        frame = cv2.resize(frame, (ancho, alto))

        cv2.imshow('Video original', frame)
        
        # Para cerrar el vídeo
        if cv2.waitKey(5) & 0xFF == 27:# Código ACII esc == 27:
            break

    cv2.destroyAllWindows()

    # Release el frame
    cap.release()


def obtener_fondo(video, ancho, alto):
    """
    Calcula el fondo estático de un vídeo <video> promediando todos sus frames.
    Devuelve la imagen del fondo redimensionada al tamaño indicado (<ancho>, <alto>).
    """
    cap = leer_video(video)

    frames = [] # Donde añadiremos todos los frames del vídeo

    # Recorremos todo el vídeo para almacenar los frames
    while(True):
        ret, frame = cap.read()
        
        # Comprobamos que se esté visualizando correctamente
        if not ret:
            print("Fin del vídeo")
            break

        # Ajustamos para que el vídeo ocupe menos
        frame = cv2.resize(frame, (ancho, alto))

        # Convertimos a float para evitar saturación en la suma
        frames.append(frame.astype(np.float32))

        # Para terminar el proceso
        if cv2.waitKey(5) & 0xFF == 27: # Código ACII esc == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

    # Calcular el promedio
    # np.mean -> hace la media de todos los frames
    # .astype -> convierte los float al formato que usa opencv (uint8)
    promedio =  np.mean(frames, axis=0).astype(np.uint8)

    # cv2.imshow("Fondo promedio", promedio) # Título que se muestra en la ventana
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("images/fondo_sin_coches.jpg", promedio) # Escribe la imagen generada en la ruta descrita
    fondo = cv2.resize(promedio, (ancho, alto)) # Redimensiona la imagen del fondo según los parametros
    return fondo


def quitar_fondo(video, fondo, ancho, alto):    
    """
    Resta el fondo estático de un vídeo para resaltar los objetos en movimiento.
    <video>: ruta del archivo de vídeo
    <fondo>: ruta de la imagen de fondo
    <ancho>, <alto>: dimensiones a las que se redimensionarán ambos
    """
    cap = leer_video(video)
    # Leemos la imagen, la redimensionamos, y la cambiamos al formato uint8 para poder trabajar con ella
    fondo = cv2.resize(cv2.imread(fondo), (ancho, alto)).astype(np.uint8)

    # Recorremos el video
    while(True):
        ret, frame = cap.read()
        
        # Comprobamos que se esté visualizando correctamente
        if not ret:
            print("Fin del vídeo")
            break

        # Ajustamos para que el vídeo coincida con las dimensiones del fondo
        frame = cv2.resize(frame, (ancho, alto))

        # Restamos
        diferencia = cv2.absdiff(frame, fondo)
        cv2.imshow('Video original', diferencia)


        # Para cerrar el vídeo
        if cv2.waitKey(5) & 0xFF == 27: # Código ACII esc == 27:
            break

    cv2.destroyAllWindows()
    cap.release()


def quitar_fondo_umbralizado(video, fondo, ancho, alto):    
    """
    Resta el fondo estático de un vídeo y umbraliza la diferencia
    para resaltar únicamente los objetos en movimiento (vehículos).
    <video>: ruta del archivo de vídeo
    <fondo>: ruta de la imagen de fondo
    <ancho>, <alto>: dimensiones a las que se redimensionarán ambos
    """
    cap = leer_video(video)

    # Leemos la imagen del fondo, la redimensionamos y la convertimos al formato uint8
    fondo = cv2.resize(cv2.imread(fondo), (ancho, alto)).astype(np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del vídeo")
            break

        frame = cv2.resize(frame, (ancho, alto))

        # Restamos el fondo
        diferencia = cv2.absdiff(frame, fondo)

        # Convertimos a escala de grises (solo necesitamos intensidad, no color)
        gris = cv2.cvtColor(diferencia, cv2.COLOR_BGR2GRAY)

        # Aplicamos un umbral para destacar las zonas con diferencias significativas
        # (ajusta 50 según la iluminación del vídeo)
        _, umbral = cv2.threshold(gris, 50, 255, cv2.THRESH_BINARY)


        # Mostramos resultados
        cv2.imshow('Diferencia', diferencia)
        cv2.imshow('Movimiento detectado', umbral)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC para salir
            break

    cv2.destroyAllWindows()
    cap.release()


def detectar_coches(video, fondo, ancho, alto, min_area):
    """
    Detecta los vehículos (blobs) en movimiento a partir del vídeo <video> y el fondo <fondo>.
    Muestra en pantalla los contornos detectados en cada frame.
    """
    cap = leer_video(video)
    fondo = cv2.resize(cv2.imread(fondo), (ancho, alto)).astype(np.uint8)

    # Definimos la ROI según el vídeo
    x1, y1, x2, y2 = 0, 125, 800, 450

    # Zona a ignorar (timestamp u overlay de texto)
    x_txt, y_txt, w_txt, h_txt = 675, 400, 30, 20  # coordenadas del recuadro de los números

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del vídeo")
            break

        frame = cv2.resize(frame, (ancho, alto))

        # Eliminar el timestamp (píxeles cambiantes de la fecha/hora)
        cv2.rectangle(frame, (x_txt, y_txt), (x_txt + w_txt, y_txt + h_txt), (0, 0, 0), -1)
        cv2.rectangle(fondo, (x_txt, y_txt), (x_txt + w_txt, y_txt + h_txt), (0, 0, 0), -1)
        #Dibuja rectángulos: (x_txt, y_txt) es la esquina superior izquierda y la otra coordenada es la
        #inferior derecha. (0, 0, 0) indica color negro y -1 que el relleno sea sólido (completo).

        diferencia = cv2.absdiff(frame, fondo)

        # Convertimos a escala de grises (solo necesitamos intensidad, no color)
        gris = cv2.cvtColor(diferencia, cv2.COLOR_BGR2GRAY)

        # Umbralizamos para quedarnos con zonas de movimiento
        _, umbral = cv2.threshold(gris, 33, 255, cv2.THRESH_BINARY)
        # cv2.threshold binariza gris: todos los píxeles con valor > 40 pasan a 255 (blanco), el resto a 0 (negro)
        # 40 es el umbral (sensibilidad): más bajo → más sensible; más alto → menos falsas detecciones
        # <_> es el umbral usado (lo ignoramos), <umbral> es la imagen binaria

        # Aplicamos operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # Crea un kernel rectangular 5×5 (matriz de 1s) que se usará para las operaciones morfológicas
        # Controla cuánto se “engrosan/recortan” regiones: mayor kernel → cambios más fuertes
        
        umbral = cv2.morphologyEx(umbral, cv2.MORPH_CLOSE, kernel)  # cierra huecos
        umbral = cv2.morphologyEx(umbral, cv2.MORPH_OPEN, kernel)   # quita ruido

        #Aplicamos la máscara de la ROI
        mask = np.zeros_like(umbral) # crea una imagen en negro del tamaño de la imagen umbral
        mask[y1:y2, x1:x2] = 255 # Pone los pixeles de la zona de interes en la imagen mask en blanco
        umbral_roi = cv2.bitwise_and(umbral, mask) # Se aplica un AND de ambas imágenes, eliminando los píxeles blancos que no están en la ROI y conservando los que sí

        # Buscamos contornos (posibles coches)
        contornos, _ = cv2.findContours(umbral_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.findContours detecta los contornos en la imagen binaria umbral_roi
        # cv2.RETR_EXTERNAL toma solo los contornos externos (ignora contornos interiores)
        # cv2.CHAIN_APPROX_SIMPLE comprime los puntos del contorno para ahorrar memoria
        # Devuelve contornos (lista de arrays con puntos) y una jerarquía (aquí ignorada con _)

        for cont in contornos:
            area = cv2.contourArea(cont) # Iteramos cada contorno y calculamos su área en píxeles
            if area > min_area:  # filtra ruido: ajustamos este umbral según el vídeo
                x, y, w, h = cv2.boundingRect(cont) # cv2.boundingRect devuelve el rectángulo mínimo alineado a los ejes que contiene el contorno
                # x, y = coordenadas de la esquina superior izquierda; w, h = ancho y alto del rectángulo
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Dibuja un rectángulo verde
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) #ROI

        # Ventanas a mostrar
        cv2.imshow('Coches detectados', frame)
        cv2.imshow('Máscara movimiento', umbral)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
