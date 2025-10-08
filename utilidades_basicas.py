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