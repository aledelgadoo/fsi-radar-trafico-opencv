import numpy as np
import cv2
from vehiculos import *

class GestorVehiculos:
    """
    Clase que gestiona todos los vehículos detectados y hace el seguimiento
    entre frames, manteniendo consistencia de IDs y estados.
    Versión con Filtro de Kalman.
    """
    
    def __init__(self, umbral_distancia=50, max_frames_perdido=10):
        """
        umbral_distancia: píxeles máximos para considerar que una detección
                          pertenece al mismo vehículo.
        max_frames_perdido: cuántos frames puede estar sin verse antes de desactivarlo.
        """
        self.vehiculos = []
        self.umbral_distancia = umbral_distancia
        self.max_frames_perdido = max_frames_perdido


    def actualizar(self, detecciones, frame, frame_num):
        """
        Actualiza el estado de todos los vehículos a partir de las nuevas detecciones.
        
        detecciones: lista de bounding boxes [(x, y, w, h), ...] detectadas en este frame.
        """
        
        # --- PASO 1: PREDECIR ---
        # Obtenemos la posición predicha para todos los vehículos activos
        # y los guardamos en un diccionario {vehiculo: pos_predicha}
        pos_predichas = {}
        for v in self.vehiculos_activos():
            pos_predichas[v] = v.predecir()

        # --- PASO 2: PREPARAR DETECCIONES ---
        # Calculamos centroides de las nuevas detecciones
        nuevos_centroides = []
        for bbox in detecciones:
            nuevos_centroides.append(self._centroide(bbox))

        # --- PASO 3: ASOCIAR Y CORREGIR (o CREAR) ---
        vehiculos_actualizados = set()

        for bbox, centroide in zip(detecciones, nuevos_centroides):
            dist_min = self.umbral_distancia
            vehiculo_asociado = None

            # Buscamos el vehículo (predicho) más cercano a esta detección
            for v, pos_predicha in pos_predichas.items():
                
                # Si este vehículo ya fue asignado a otra detección, saltar
                if v in vehiculos_actualizados:
                    continue
                    
                dist = np.linalg.norm(np.array(pos_predicha) - np.array(centroide))
                
                if dist < dist_min:
                    dist_min = dist
                    vehiculo_asociado = v
            
            if vehiculo_asociado:
                # --- CORREGIR ---
                # Asociado: Corregimos el filtro con la medición real
                vehiculo_asociado.corregir(centroide)
                # Actualizamos su BBox y reseteamos contadores
                vehiculo_asociado.bbox = bbox
                vehiculo_asociado.frames_perdido = 0
                vehiculo_asociado.frames_activo += 1
                vehiculos_actualizados.add(vehiculo_asociado)
            
            else:
                # --- CREAR ---
                # No asociado: Es un vehículo nuevo
                nuevo = Coche(centroide, bbox, frame_num)
                self.vehiculos.append(nuevo)
        
        # --- PASO 4: MARCAR PERDIDOS (Oclusión) ---
        # Para los vehículos que predijimos pero que NO fueron actualizados
        # (ej. ocultos tras una farola)
        for v in pos_predichas.keys():
            if v not in vehiculos_actualizados:
                v.marcar_perdido(self.max_frames_perdido)
        
        
        # --- PASO 5: LIMPIEZA DE INACTIVOS ANTIGUOS ---
        vehiculos_activos_lista = []
        for v in self.vehiculos:
            # Mantenemos los activos O los inactivos recientes
            # (les damos un "tiempo de gracia" antes de borrarlos)
            if v.activo or v.frames_perdido < 1000:
                vehiculos_activos_lista.append(v)
        self.vehiculos = vehiculos_activos_lista


    def _centroide(self, bbox):
        """Calcula el centroide de un bounding box (x, y, w, h)."""
        x, y, w, h = bbox
        return (x + w / 2, y + h / 2)


    def vehiculos_activos(self):
        """Devuelve solo los vehículos activos."""
        vehiculos_activos_lista = []
        for v in self.vehiculos:
            if v.activo:
                vehiculos_activos_lista.append(v)
        return vehiculos_activos_lista