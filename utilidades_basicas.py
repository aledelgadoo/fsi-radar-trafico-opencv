import cv2
import numpy as np
import matplotlib.pyplot as plt


def leer_video(video):
    cap = cv2.VideoCapture(video)

    # Mensaje por si no podemos acceder al archivo
    if not cap.isOpened():
        print("Error!")
        exit()
    
    return cap


def visualizar_video(video):
    cap = leer_video(video)

    # Visualizar el vídeo
    while(True):
        
        ret, frame = cap.read()
        
        # Comprobamos que se esté visualizando correctamente
        if not ret:
            print("Fin del vídeo")
            break

        # Ajustamos para que el vídeo ocupe menos
        frame = cv2.resize(frame, (1000, 700))

        cv2.imshow('Video original', frame)
        
        # Para cerrar el vídeo
        if cv2.waitKey(5) & 0xFF == 27:# Código ACII esc == 27:
            break

    cv2.destroyAllWindows()

    # Release el frame
    cap.release()

def obtener_fondo(video):
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
        frame = cv2.resize(frame, (400, 300))

        # Convertimos a float para evitar saturación en la suma
        frames.append(frame.astype(np.float32))

        # Para cerrar el vídeo
        if cv2.waitKey(5) & 0xFF == 27: # Código ACII esc == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

    # Calcular el promedio
    # np.mean -> hace la media de todos los frames
    # .astype -> convierte los float al formato que usa opencv (uint8)
    promedio =  np.mean(frames, axis=0).astype(np.uint8)

    cv2.imshow("Fondo promedio", promedio) # Título que se muestra en la ventana
    cv2.imwrite("images/fondo_sin_coches.jpg", promedio) # Escribe la imagen generada en la ruta descrita
    cv2.waitKey(0)
    cv2.destroyAllWindows()