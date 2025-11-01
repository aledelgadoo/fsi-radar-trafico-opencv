import cv2
import numpy as np
import matplotlib.pyplot as plt
from funcionesV1 import *
from funcionesV2 import *

def main():
    video = 'images/trafico.mp4'
    # Asegúrate que el nombre del fondo coincide con el que generaste
    fondo = 'images/(trafico)-fondo_sin_coches.jpg' 
    
    # --- Panel de control de parámetros (con Kalman) ---
    p_escala = 0.5              
    p_umbral_sensibilidad = 30  
    p_min_area_base = 230       
    p_kernel_size_base = 7      
    p_umbral_dist_base = 50     # Dist. máx. para asociar (Detección vs Predicción)
    p_max_frames_perdido = 10   # Paciencia para oclusión
    p_frames_confirmacion = 5   
    
    
    # --- Llamada a la función V2 (AHORA MÁS CORTA) ---
    detectar_cochesV2(
        video, 
        fondo, 
        escala=p_escala,
        umbral_sensibilidad=p_umbral_sensibilidad,
        min_area_base=p_min_area_base,
        kernel_size_base=p_kernel_size_base,
        umbral_dist_base=p_umbral_dist_base,
        max_frames_perdido=p_max_frames_perdido,
        frames_para_confirmar=p_frames_confirmacion,
    )

    # --- Llamadas antiguas (comentadas) ---
    # (Tu código de V1...)
    # obtener_fondo_cli(video) # Para (re)generar el fondo

if __name__ == "__main__":
    main()

