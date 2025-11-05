import numpy as np
from funcionesV1 import *
from funcionesV2 import *

def main():
    p_ruta_video = 'images/trafico.mp4'
    p_ruta_fondo = 'images/(trafico)-fondo_sin_coches.jpg' 
    
    # --- Panel de control de parámetros ---
    p_escala = 0.5              
    p_umbral_sensibilidad = 30  
    p_umbral_fusion_base = 60
    p_min_area_base = 250       
    p_kernel_size_base = 7      
    p_umbral_dist_base = 50     # Dist. máx. para asociar (Detección vs Predicción)
    p_max_frames_perdido = 10   # Paciencia para oclusión
    p_frames_confirmacion = 5   
    p_roi_base = [280, 965, 0, 1920]
    p_metodo_fondo = 'estatico' # 'estatico' (con imagen) o 'dinamico' (con MOG2)
    p_frames_calentamiento_mog2 = 100 # Cuántos frames "ignorar" al inicio para que MOG2 aprenda el fondo
    p_orientacion_via = 'vertical' # 'vertical' (suben / bajan) o 'horizontal' (izq. / der.)
    p_factor_perspectiva_max = 10

    p_filtro_sentido = None
    p_mostrar_texto_velocidad = True
    p_mostrar_texto_sentido = True
    p_mostrar_id = True
    p_mostrar_roi = True
    p_colorear_por = 'velocidad' # Opciones: None, 'sentido', 'tipo', 'velocidad'
    p_vel_min_color = 4 # La velocidad MÍNIMA para empezar el gradiente (se verá Azul)
    p_vel_max_color = 20  # La velocidad MÁXIMA (se verá Rojo)
    p_pixeles_por_metro = 37.1

    p_mostrar_contador_activos = True
    p_mostrar_contador_historico = True
    p_mostrar_contador_subiendo = True
    p_mostrar_contador_bajando = True
    p_mostrar_contador_motos = True
    p_mostrar_contador_coches = True
    p_mostrar_contador_camiones = True
    p_mostrar_tipo_coche = True
    
    p_area_moto_max_base = 1000   # Área máxima para ser 'Moto'
    p_area_coche_max_base = 10000 # Área máxima para ser 'Coche'
    # (Lo que supere esto, será 'Camion')
    
    # --- Llamada a la función V2 ---
    detectar_cochesV2(
        ruta_video=p_ruta_video, 
        ruta_fondo=p_ruta_fondo, 
        escala=p_escala,
        roi_base=p_roi_base,
        umbral_sensibilidad=p_umbral_sensibilidad,
        umbral_fusion_base=p_umbral_fusion_base,
        min_area_base=p_min_area_base,
        kernel_size_base=p_kernel_size_base,
        umbral_dist_base=p_umbral_dist_base,
        max_frames_perdido=p_max_frames_perdido,
        frames_para_confirmar=p_frames_confirmacion,
        metodo_fondo=p_metodo_fondo,
        frames_calentamiento=p_frames_calentamiento_mog2,
        orientacion_via=p_orientacion_via,
        factor_perspectiva_max=p_factor_perspectiva_max,
        pixeles_por_metro=p_pixeles_por_metro,

        filtro_sentido=p_filtro_sentido,
        mostrar_texto_velocidad=p_mostrar_texto_velocidad,
        mostrar_texto_sentido=p_mostrar_texto_sentido,
        mostrar_id=p_mostrar_id,
        mostrar_roi=p_mostrar_roi,
        colorear_por=p_colorear_por,
        vel_min_color=p_vel_min_color,
        vel_max_color=p_vel_max_color,

        mostrar_contador_activos=p_mostrar_contador_activos,
        mostrar_contador_historico=p_mostrar_contador_historico,
        mostrar_contador_subiendo=p_mostrar_contador_subiendo,
        mostrar_contador_bajando=p_mostrar_contador_bajando,

        area_moto_max_base=p_area_moto_max_base,
        area_coche_max_base=p_area_coche_max_base,
        mostrar_contador_motos=p_mostrar_contador_motos,
        mostrar_contador_coches=p_mostrar_contador_coches,
        mostrar_contador_camiones=p_mostrar_contador_camiones,
        mostrar_tipo_coche=p_mostrar_tipo_coche
    )

    # --- Llamadas antiguas ---
    # obtener_fondo(ruta_video) # Para (re)generar el fondo

def probar_trafico2():
    detectar_cochesV2(
        ruta_video='images/trafico2.mp4', # Usa la variable 'video' definida arriba
        ruta_fondo='images/(trafico2)-fondo_sin_coches.jpg', # Usa la variable 'fondo' definida arriba
        escala=0.5,
        roi_base=[450, 1080, 0, 1920],
        umbral_sensibilidad=30,
        min_area_base=3500,
        kernel_size_base=20,
        umbral_dist_base=50,
        max_frames_perdido=20,
        frames_para_confirmar=8,
        metodo_fondo='estatico',
        frames_calentamiento=100,
        factor_perspectiva_max=20,
        pixeles_por_metro=55,
        filtro_sentido=None,
        mostrar_texto_velocidad=True,
        mostrar_texto_sentido=True,
        mostrar_id=True,
        mostrar_roi=True,
        mostrar_contador_activos=True,
        mostrar_contador_historico=True,
        mostrar_contador_subiendo=True,
        mostrar_contador_bajando=True,
        area_moto_max_base=5000,
        area_coche_max_base=25000,
        mostrar_contador_motos=True,
        mostrar_contador_coches=True,
        mostrar_contador_camiones=True,
        mostrar_tipo_coche=True
    )

if __name__ == "__main__":
    probar_trafico2()