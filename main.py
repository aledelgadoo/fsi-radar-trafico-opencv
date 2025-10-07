import cv2
import numpy as np
import matplotlib.pyplot as plt
from utilidades_basicas import *

def main():
    video = 'images/trafico.mp4'
    # visualizar_video(video)
    # fondo = obtener_fondo(video)
    fondo = 'images/fondo_sin_coches.jpg'
    quitar_fondo(video, fondo)


if __name__ == "__main__":
    main()

