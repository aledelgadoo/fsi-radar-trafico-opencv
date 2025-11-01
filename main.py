import cv2
import numpy as np
import matplotlib.pyplot as plt
from funcionesV1 import *
from funcionesV2 import *

def main():
    video = 'images/trafico.mp4'
    # Asegúrate que el nombre del fondo coincide con el que generaste
    fondo = 'images/(trafico)-fondo_sin_coches.jpg' 
    
    # --- NUEVO: Panel de control de parámetros ---
    p_escala = 0.7              # 0.5=50%, 1.0=100%. Afecta a todo lo demás.
    
    # Parámetros de Detección (Máscara)
    p_umbral_sensibilidad = 50  # (Default: 50) Más bajo = más sensible (más ruido).
    p_min_area_base = 60        # (Default: 60) Área mínima (para escala 1.0).
    p_kernel_size_base = 5      # (Default: 5) Tamaño del kernel (para escala 1.0).

    # Parámetros de Tracking (IDs)
    p_umbral_dist_base = 50     # (Default: 50) Dist. máx. para asociar coche.
    p_max_frames_perdido = 10   # (Default: 10) Paciencia antes de borrar ID.
    
    
    # --- Llamada a la función V2 con todos los parámetros ---
    detectar_cochesV2(
        video, 
        fondo, 
        escala=p_escala,
        umbral_sensibilidad=p_umbral_sensibilidad,
        min_area_base=p_min_area_base,
        kernel_size_base=p_kernel_size_base,
        umbral_dist_base=p_umbral_dist_base,
        max_frames_perdido=p_max_frames_perdido
    )

    # --- Llamadas antiguas (comentadas) ---
    # (Tu código de V1...)
    # obtener_fondo_cli(video) # Para (re)generar el fondo

if __name__ == "__main__":
    main()

