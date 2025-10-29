import cv2
import numpy as np
import matplotlib.pyplot as plt
from funcionesV1 import *
from funcionesV2 import *

def main():
    video = 'images/trafico.mp4'
    fondo = 'images/fondo_sin_coches.jpg'
    ancho, alto = (800, 450) # Parámetros para la redimensión
    min_area = 50
    linea_y = 250
    min_dist = 20
    # visualizar_video(video, ancho, alto)
    # obtener_fondo(video, ancho, alto)
    # quitar_fondo(video, fondo, ancho, alto)
    # quitar_fondo_umbralizado(video, fondo, ancho, alto)
    # detectar_coches(video, fondo, ancho, alto, min_area)
    # contar_coches(video, fondo, ancho, alto, linea_y, min_area, min_dist)
    detectar_cochesV2(video, fondo, ancho, alto)

if __name__ == "__main__":
    main()

