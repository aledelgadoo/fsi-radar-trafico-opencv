from abc import ABC, abstractmethod
import numpy as np
import math
import cv2

from abc import ABC, abstractmethod
import numpy as np
import cv2

class Vehiculo(ABC):
    """
    Clase base abstracta.
    (La dejamos por si en el futuro queremos 'Motos' o 'Camiones' 
    con físicas distintas, pero ahora no la usaremos directamente).
    """
    _next_id = 0

    def predecir(self):
        pass

    def corregir(self, medicion):
        pass

class Coche(Vehiculo):
    """
    Representa un vehículo individual.
    Ahora incluye un Filtro de Kalman para predecir su movimiento 
    y estimar su velocidad.
    """
    def __init__(self, centroide, bbox, frame_num):
        self.id = Vehiculo._next_id
        Vehiculo._next_id += 1

        self.bbox = bbox
        self.frame_num = frame_num
        self.activo = True
        self.frames_perdido = 0
        self.frames_activo = 1
        self.ya_contado_salida = False # Dejamos esto por si acaso

        # --- INICIO DEL FILTRO DE KALMAN ---
        # 1. Creamos el filtro
        # 4 -> Número de estados [x, y, vx, vy] (posición y velocidad)
        # 2 -> Número de mediciones [x, y] (solo medimos la posición)
        self.kalman = cv2.KalmanFilter(4, 2)
        
        # 2. Definimos el estado inicial
        # [x, y, 0, 0] -> Posición inicial y velocidad 0
        self.kalman.statePost = np.array([centroide[0], centroide[1], 0, 0], dtype=np.float32)
        self.kalman.statePre = np.array([centroide[0], centroide[1], 0, 0], dtype=np.float32)

        # 3. Matriz de Transición (A)
        # Cómo el estado cambia de t a t+1. (dt = 1 frame)
        # x_t+1 = x_t + vx_t
        # y_t+1 = y_t + vy_t
        # vx_t+1 = vx_t
        # vy_t+1 = vy_t
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # 4. Matriz de Medición (H)
        # Cómo pasamos del estado (4D) a la medición (2D)
        # [x_med] = [1, 0, 0, 0] * [x, y, vx, vy].T
        # [y_med] = [0, 1, 0, 0] * [x, y, vx, vy].T
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # 5. Ruido del Proceso (Q)
        # Incertidumbre en la predicción (aceleración, giros...)
        # Le damos más incertidumbre a la velocidad
        self.kalman.processNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 5, 0],  # Incertidumbre en vx
            [0, 0, 0, 5]   # Incertidumbre en vy
        ], dtype=np.float32) * 1e-3 # Ajustar este 1e-3 (0.001)

        # 6. Ruido de la Medición (R)
        # Incertidumbre de nuestro detector (blob)
        # Confiamos bastante en nuestra detección
        self.kalman.measurementNoiseCov = np.array([
            [1, 0],
            [0, 1]
        ], dtype=np.float32) * 1e-1 # Ajustar este 1e-1 (0.1)
        
        # --- FIN DEL FILTRO DE KALMAN ---

    def predecir(self):
        """Devuelve la posición [x, y] predicha por el filtro."""
        self.kalman.predict()
        # El estado predicho es [x, y, vx, vy]
        return self.kalman.statePre[0:2].astype(int)

    def corregir(self, medicion):
        """Corrige el filtro con una medición real [x, y]."""
        # La medición debe ser un vector columna
        medicion_np = np.array([medicion[0], medicion[1]], dtype=np.float32).reshape(2, 1)
        self.kalman.correct(medicion_np)

    @property
    def centroide(self):
        """Devuelve el centroide (x, y) actual del filtro."""
        return self.kalman.statePost[0:2].astype(int)
    
    @property
    def velocidad(self):
        """Devuelve la velocidad (vx, vy) actual del filtro."""
        return self.kalman.statePost[2:4]

    def marcar_perdido(self, limite):
        """Incrementa el contador de frames perdidos."""
        self.frames_perdido += 1
        if self.frames_perdido > limite:
            self.activo = False
    
    # --- Funciones que ya no usaremos (historial, distancia_a, etc.) ---
    # (Las borramos para que no haya confusión)