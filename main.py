import cv2
import numpy as np
from funcionesV1 import *
from funcionesV2 import *

def main():
    p_ruta_video = 'images/trafico.mp4'
    p_ruta_fondo = 'images/(trafico)-fondo_sin_coches.jpg' 
    
    # --- Panel de control de parámetros ---
    p_escala = 0.3              
    p_umbral_sensibilidad = 30  
    p_min_area_base = 230       
    p_kernel_size_base = 7      
    p_umbral_dist_base = 50     # Dist. máx. para asociar (Detección vs Predicción)
    p_max_frames_perdido = 10   # Paciencia para oclusión
    p_frames_confirmacion = 5   
    p_roi_base = [280, 965, 0, 1920]

    p_filtro_sentido = 'BAJA'
    p_mostrar_texto_velocidad = True
    p_mostrar_texto_sentido = True
    p_mostrar_id = True
    p_mostrar_roi = False

    p_mostrar_contador_activos = True
    p_mostrar_contador_historico = True
    p_mostrar_contador_subiendo = True
    p_mostrar_contador_bajando = True
    
    
    # --- Llamada a la función V2 ---
    detectar_cochesV2(
        ruta_video=p_ruta_video, 
        ruta_fondo=p_ruta_fondo, 
        escala=p_escala,
        roi_base=p_roi_base,
        umbral_sensibilidad=p_umbral_sensibilidad,
        min_area_base=p_min_area_base,
        kernel_size_base=p_kernel_size_base,
        umbral_dist_base=p_umbral_dist_base,
        max_frames_perdido=p_max_frames_perdido,
        frames_para_confirmar=p_frames_confirmacion,

        filtro_sentido=p_filtro_sentido,
        mostrar_texto_velocidad=p_mostrar_texto_velocidad,
        mostrar_texto_sentido=p_mostrar_texto_sentido,
        mostrar_id=p_mostrar_id,
        mostrar_roi=p_mostrar_roi,

        mostrar_contador_activos=p_mostrar_contador_activos,
        mostrar_contador_historico=p_mostrar_contador_historico,
        mostrar_contador_subiendo=p_mostrar_contador_subiendo,
        mostrar_contador_bajando=p_mostrar_contador_bajando
    )

    # --- Llamadas antiguas ---
    # obtener_fondo(ruta_video) # Para (re)generar el fondo

if __name__ == "__main__":
    main()