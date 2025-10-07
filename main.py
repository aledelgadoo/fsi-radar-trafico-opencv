import cv2
import numpy as np
import matplotlib.pyplot as plt
from utilidades_basicas import *

def main():
    video = 'images/trafico.mp4'
    # visualizar_video(video)
    obtener_fondo(video)

if __name__ == "__main__":
    main()

