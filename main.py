import cv2
import numpy as np
import matplotlib.pyplot as plt
from utilidades_basicas import *

def main():
    video = 'images/trafico.mp4'
    fondo = 'images/fondo_sin_coches.jpg'
    ancho, alto = (800, 450) # Parámetros para la redimensión
    min_area = 40
    # visualizar_video(video, ancho, alto)
    # obtener_fondo(video, ancho, alto)
    # quitar_fondo(video, fondo, ancho, alto)
    # quitar_fondo_umbralizado(video, fondo, ancho, alto)
    detectar_coches(video, fondo, ancho, alto, min_area)

if __name__ == "__main__":
    main()

