import numpy as np
import cv2
from vehiculos import *

class GestorVehiculos:
    """
    Clase que gestiona todos los vehículos detectados y hace el seguimiento
    entre frames, manteniendo consistencia de IDs y estados.
    """
    
    def __init__(self, umbral_distancia=50, max_frames_perdido=10):
        """
        umbral_distancia: píxeles máximos para considerar que una detección
                          pertenece al mismo vehículo.
        max_frames_perdido: cuántos frames puede estar sin verse antes de desactivarlo.
        """
        self.vehiculos = []  # lista de objetos hijos de vehiculos activos
        self.umbral_distancia = umbral_distancia
        self.max_frames_perdido = max_frames_perdido


    def actualizar(self, detecciones, frame, frame_num):
        """
        Actualiza el estado de todos los vehículos a partir de las nuevas detecciones.
        
        detecciones: lista de bounding boxes [(x, y, w, h), ...] detectadas en este frame.
        frame: imagen actual (para recortar coches si se quiere guardar frame_img).
        frame_num: número del frame actual.
        """
        # Calculamos centroides de las nuevas detecciones
        nuevos_centroides =  []
        for bbox in detecciones:
            nuevos_centroides.append(self._centroide(bbox))

        # Marcamos todos los vehículos como no actualizados inicialmente
        actualizados = set()

        # Intentamos asociar cada nueva detección a un vehículo existente
        for bbox, centroide in zip(detecciones, nuevos_centroides):
            vehiculo_asociado = None
            distancia_min = float('inf') # inicializamos la variable que usaremos más adelante

            for vehiculo in self.vehiculos:
                if not vehiculo.activo:
                    continue
                    
                d = vehiculo.distancia_a(centroide)
                if d < self.umbral_distancia and d < distancia_min:
                    distancia_min = d # si dos coches están lo suficientemente cerca, nos quedamos con el que más cerca esté
                    vehiculo_asociado = vehiculo
            
            if vehiculo_asociado:
                # Actualizamos vehículo existente
                x, y, w, h = bbox
                if frame is not None: # control de errores. ChatGPT recomendó escribir is not None
                    frame_crop = frame_crop = frame[y:y+h, x:x+w]
                else:
                    frame_crop = None
                vehiculo_asociado.actualizar(centroide, bbox, frame_num) # actualizar de Vehiculos no de Gestor
                actualizados.add(vehiculo_asociado)
            
            else:
                # Creamos nuevo vehículo
                x, y, w, h = bbox
                if frame is not None: # control de errores. ChatGPT recomendó escribir is not None
                    frame_crop = frame_crop = frame[y:y+h, x:x+w]
                else:
                    frame_crop = None
                nuevo = Coche(centroide, bbox, frame_crop, frame_num)
                self.vehiculos.append(nuevo)
        
        # Para los no actualizados, marcamos como perdidos e inactivos
        for vehiculo in self.vehiculos:
            if vehiculo not in actualizados and vehiculo.activo:
                vehiculo.marcar_perdido(self.max_frames_perdido)
        
        
        # --- NUEVO: limpieza de inactivos antiguos ---
        vehiculos_activos = []
        for v in self.vehiculos:
            if v.activo or v.frames_perdido < 1000:
                vehiculos_activos.append(v)
        self.vehiculos = vehiculos_activos


    def _centroide(self, bbox):
        """Calcula el centroide de un bounding box (x, y, w, h)."""
        x, y, w, h = bbox
        return (x + w / 2, y + h / 2)


    def vehiculos_activos(self):
        """Devuelve solo los vehículos activos."""
        vehiculos_activos = []              # Creamos una lista vacía
        for v in self.vehiculos:            # Recorremos todos los vehículos
            if v.activo:                    # Si el vehículo está activo
                vehiculos_activos.append(v) # lo añadimos a la lista
        return vehiculos_activos  