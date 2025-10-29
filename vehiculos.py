from abc import ABC, abstractmethod
import numpy as np
import math
import cv2

class Vehiculo(ABC):
    """
    Clase base abstracta para representar un vehículo detectado en el vídeo.
    Guarda su posición (centroide), bounding box, frame donde se detectó, 
    y tiene lógica para calcular distancia o determinar si sigue siendo el mismo.
    """
    _next_id = 0  # Atributo de clase para manejar las ids

    def __init__(self, centroide, bbox, frame_img, frame_num):
        self.id = Vehiculo._next_id   # id del vehiculo al instanciar
        Vehiculo._next_id += 1        # Manejamos las ids con el atributo estático

        self.centroide = np.array(centroide, dtype=float)  # (x, y)
        self.bbox = bbox                 # (x, y, w, h)
        self.frame_img = frame_img       # Recorte del vehículo en el frame
        self.frame_num = frame_num       # Frame donde se detectó

        self.activo = True               # True mientras se sigue detectando
        self.frames_perdido = 0          # Contador de frames consecutivos sin verlo
        self.historial = [centroide]     # Guarda trayectoria para velocidad o debug futuro

    def distancia_a(self, otro_centroide):
        """Calcula la distancia euclídea entre el centro actual y otro."""
        return np.linalg.norm(self.centroide - np.array(otro_centroide))

    def actualizar(self, nuevo_centroide, nueva_bbox, frame_num):
        """
        Actualiza la posición del vehículo con una nueva detección.
        Reinicia el contador de pérdida y añade la posición al historial.
        """
        self.centroide = np.array(nuevo_centroide, dtype=float)
        self.bbox = nueva_bbox
        self.frame_num = frame_num
        self.frames_perdido = 0
        self.historial.append(nuevo_centroide)

    def marcar_perdido(self, limite):
        """Incrementa el contador de frames perdidos."""
        self.frames_perdido += 1
        if self.frames_perdido > limite:  # se puede ajustar
            self.activo = False

    def es_mismo_vehiculo(self, otro_centroide, umbral_distancia=50):
        """
        Determina si una nueva detección pertenece al mismo vehículo
        en función de la distancia entre centroides.
        """
        return self.distancia_a(otro_centroide) < umbral_distancia
    
    def tipo(self):
        """Devuelve el tipo de vehículo (debe implementarse en subclases)."""
        pass


class Coche(Vehiculo):
    """
    Subclase que representa específicamente un coche.
    Por ahora hereda todo de Vehiculo, pero en el futuro puede
    añadir propiedades específicas (tipo, tamaño, velocidad media, etc.)
    """
    def __init__(self, centroide, bbox, frame_img=None, frame_num=None):
        super().__init__(centroide, bbox, frame_img, frame_num)
        self.tipo = "Coche"

    def tipo(self):
        return self.tipo
