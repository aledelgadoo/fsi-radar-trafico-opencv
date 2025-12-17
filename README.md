# 🚗 Análisis de Tráfico con Visión Artificial y OpenCV
**Autores:** Alejandro Delgado y Tomás Santana  
**Asignatura:** Fundamentos de los Sistemas Inteligentes *(Práctica 1)*  
**Universidad de Las Palmas de Gran Canaria - Curso 25/26**  
**Versión:** v1.0  

---

## 1. Introducción y Objetivos
El presente proyecto tiene como objetivo el desarrollo de un sistema de visión artificial capaz de detectar, contar, clasificar y estimar la velocidad de vehículos en vías de tráfico. La implementación se ha realizado en Python utilizando la librería **OpenCV** para el procesamiento de imagen y **Tkinter** para la interfaz de usuario, siguiendo una metodología de desarrollo incremental que culminó en una refactorización modular.

## 2. Metodología y Evolución del Desarrollo

El desarrollo del sistema ha seguido un enfoque iterativo dividido en cuatro fases claramente diferenciadas, que permitieron evolucionar desde pruebas de concepto básicas hasta una aplicación robusta y estructurada.

### Fase 1: Prototipado Inicial (`funcionesV1.py`)
En la etapa inicial, se desarrollaron scripts procedimentales para validar las técnicas básicas de visión por computador:
* **Extracción de Fondo:** Implementación del algoritmo de promedio temporal (`obtener_fondo`) para generar un modelo estático del fondo vacío, eliminando los vehículos en movimiento de la escena base.
* **Detección Básica:** Uso de la diferencia absoluta (`cv2.absdiff`) y umbralización binaria para detectar movimiento y validar la obtención de Regiones de Interés (ROIs).
* **Enfoque Inicial (Línea Virtual):** Se implementó inicialmente un método simple de conteo basado en líneas virtuales. Esta técnica registraba un vehículo cada vez que el centroide de la detección cruzaba una coordenada de píxel predefinida. Esta solución fue rápida para validar la detección de movimiento, pero demostró ser no escalable para el resto de requisitos del proyecto (estimación de velocidad, clasificación y manejo de oclusiones). Por sugerencia del profesorado, se determinó que el enfoque de línea debía ser reemplazado por un sistema basado en persistencia de identidad (tracking).
  
* *Limitación:* Estas funciones sirvieron como prueba de concepto pero carecían de persistencia temporal (*tracking*), lo que provocaba conteos erróneos ante parpadeos o detenciones.

### Fase 2: Arquitectura Orientada a Objetos
Para resolver los problemas de pérdida de identidad y dotar al sistema de "memoria", se migró el núcleo lógico hacia un paradigma de Orientación a Objetos:
* **Modelo `Vehiculo` (`vehiculos.py`):** Se encapsuló el estado de cada coche en un objeto. La mejora crítica fue la integración del **Filtro de Kalman** (`cv2.KalmanFilter`). Este filtro permite predecir la posición futura del vehículo y suavizar su trayectoria, siendo esencial para obtener una estimación estable de la velocidad y evitar saltos en la detección.
* **Controlador `GestorVehiculos` (`gestor_vehiculos.py`):** Se desarrolló un gestor de identidades capaz de asociar las detecciones de cada *frame* con los vehículos existentes, minimizando la distancia euclidiana. Además, maneja oclusiones temporales mediante un sistema de "paciencia" (`max_frames_perdido`), permitiendo recuperar la identidad de un coche tras pasar tras un obstáculo.

### Fase 3: Lógica Avanzada (`funcionesV2.py`)
Sobre la base de objetos, se desarrollaron algoritmos complejos para cumplir los requisitos funcionales de la práctica:
* **Corrección de Fragmentación:** Se detectó que vehículos grandes (camiones) se dividían en múltiples detecciones. Se implementó el algoritmo `fusionar_detecciones_cercanas` para agrupar detecciones próximas en una sola entidad.
* **Clasificación y Física:** Implementación de lógica para diferenciar entre **Motos, Coches y Camiones** analizando el área del contorno y su relación de aspecto (*aspect ratio*). Cálculo de la velocidad vectorial y determinación del sentido de la marcha (Subiendo/Bajando, Izquierda/Derecha).
* **Gestión de Atascos:** Integración del sustractor de fondo dinámico **MOG2**, permitiendo al sistema adaptarse a cambios de luz y gestionar vehículos que se detienen (incorporándolos al fondo temporalmente).
* **Corrección de Velocidad por Perspectiva:** Se incluyó un parámetro de `factor_perspectiva_max` y la lógica asociada, que aplica una **interpolación lineal** a la velocidad para corregir el sesgo de la cámara. Esto asegura que los vehículos lejanos (que visualmente se mueven menos píxeles) reporten una velocidad coherente con los vehículos cercanos.

### Fase 4: Refactorización e Integración Final (`functions.py`)
En la etapa final del desarrollo, se realizó una limpieza y unificación del código (**Refactoring**) para mejorar la calidad del software.
* **Unificación de Módulos:** Se fusionaron las primitivas robustas de la Fase 1 (lectura y preprocesamiento) con la lógica avanzada de la Fase 3 en un único módulo consolidado llamado **`functions.py`**.
* **Beneficio:** Esta reestructuración eliminó redundancias, centralizó toda la lógica de visión computacional en un solo fichero y simplificó las dependencias del proyecto.

---

### Gestión del Flujo de Trabajo y Control de Versiones
Para garantizar un desarrollo ordenado y colaborativo, se implementó una estrategia de control de versiones basada en GitFlow simplificado. El flujo de trabajo se estructuró de la siguiente manera:

* **Rama de Desarrollo (`dev`):** Actuó como el eje central de integración. Todo el código estable se unificaba en esta rama.
* **Ramas de Funcionalidad (`feat/...`):** Cada nueva característica o módulo (ej. `feat/filtro-kalman`, `feat/interfaz-tkinter`) se desarrolló en una rama aislada creada a partir de `dev`.
* **Pull Requests (PR):** La fusión de las ramas `feat` hacia `dev` se realizó exclusivamente mediante *Pull Requests*. Esto permitió revisar el código por ambos miembros antes de integrarlo, evitando conflictos y asegurando que la rama de desarrollo se mantuviera funcional en todo momento.

## 3. Aporte Personal: Interfaz Gráfica de Usuario (GUI)

**Decisión de arquitectura (Rendimiento):** Inicialmente, se prototipó una GUI con Streamlit. Sin embargo, debido a la latencia inherente de la compresión y transmisión de vídeo en tiempo real en entornos web, se tomó la decisión de migrar la interfaz a la tecnología nativa de Tkinter. Esto eliminó el lag y garantizó la reproducción fluida del vídeo a tiempo real, algo crítico para una herramienta de visión por computador.
Como valor añadido significativo al proyecto, se ha desarrollado una aplicación de escritorio completa utilizando la librería **Tkinter**. El objetivo de este aporte es transformar el script de detección en una herramienta de software usable por un usuario final sin conocimientos de programación.

Las características principales de la interfaz (`app-tkinter.py`) incluyen:

* **Carga de Vídeos Intuitiva:** Permite al usuario seleccionar archivos de vídeo locales mediante un explorador de archivos nativo.
* **Panel de Configuración Dinámica:** Se ha diseñado un panel de control lateral que permite ajustar en tiempo real los parámetros críticos del algoritmo sin reiniciar la aplicación:
    * Ajuste de sensibilidad de detección y áreas mínimas/máximas para filtrar ruido.
    * Selección del método de fondo (Estático vs Dinámico MOG2).
    * Configuración de la orientación de la vía (Vertical/Horizontal).
* **Visualización Parametrizable:** Controles (*Checkboxes*) para activar o desactivar capas de información sobre el vídeo (mostrar/ocultar IDs, vectores de velocidad, contadores globales, cajas delimitadoras, etc.).
* **Persistencia de Parámetros (Presets):** Se implementó la funcionalidad para Guardar y **Cargar configuraciones en ficheros JSON**. Esto permite al usuario guardar un conjunto óptimo de parámetros (sensibilidad, áreas, ROI, etc.) para un vídeo específico y reutilizarlo con un solo clic, sin tener que reajustar los sliders manualmente.

Esta interfaz actúa como orquestador, conectando la entrada del usuario con la lógica del módulo `functions.py` y el `GestorVehiculos`, haciendo del sistema una solución flexible y adaptable a diferentes escenarios de tráfico.

## 4. Conclusiones
El sistema desarrollado culmina en una herramienta funcional y altamente configurable para el análisis de tráfico. La combinación de la arquitectura Orientada a Objetos con el Filtro de Kalman ha resuelto con éxito los problemas de pérdida de identidad y ofrece un tracking robusto, permitiendo una clasificación y estimación de velocidad coherente (gracias a la corrección de perspectiva).

No obstante, es fundamental reconocer las limitaciones inherentes a los métodos clásicos de Visión por Computador (OpenCV). Al depender únicamente de la resta de fondo (`absdiff`/`MOG2`) y el análisis de contornos, el sistema puede enfrentar dificultades significativas bajo ciertas condiciones:

* **Condiciones Ambientales:** Baja luz, lluvia, niebla o reflejos solares pueden degradar seriamente la calidad de la detección.
* **Oclusiones y Congestión Extrema:** En situaciones de tráfico muy denso, la superposición de objetos (`blobs`) puede desafiar incluso al Filtro de Kalman, afectando la estabilidad del tracking.

Si bien la solución actual cumple con todos los objetivos y demuestra una implementación sólida de los principios de Sistemas Inteligentes, para alcanzar una precisión "cercana a la perfección" en un entorno real y diverso, el paso siguiente en la evolución del proyecto sería migrar la fase de detección a modelos de Deep Learning (como YOLO o SSD).

<br>

<p align="center">
  <img width="50%" alt="image" src="https://github.com/user-attachments/assets/b4c47d04-6ee6-4bc7-af93-7d05c473e2d6" />
</p>
