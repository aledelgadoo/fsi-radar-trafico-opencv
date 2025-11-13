import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import json

# --- Importamos el backend ---
from gestor_vehiculos import GestorVehiculos
from vehiculos import Vehiculo
from functions import *

class RadarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Radar de Vehículos FSI - ULPGC (v0.2 Tkinter)")
        self.root.geometry("1500x850") 

        # --- Variables de estado ---
        self.processing_active = False
        self.cap = None
        self.gestor = None
        self.sustractor_fondo = None
        self.frame_num = 0
        self.fondo_redimensionado = None
        self.mask_roi = None
        self.roi_escalada = None
        self.fps = 30.0
        self.video_path = None
        self.fondo_path = None
        self.new_size = (0, 0) # Para evitar errores

        self.params_vars = {} 
        self.params = {} 

        # --- Layout Principal ---
        self.frame_controles = ttk.Frame(self.root, width=450)
        self.frame_controles.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self.frame_principal = ttk.Frame(self.root)
        self.frame_principal.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.setup_panel_carga()
        self.setup_panel_tabs() 
        self.setup_panel_video() 
        
        # --- Atributos para los parámetros escalados (se calculan en start_processing) ---
        self.min_area_escalada = 0
        self.kernel_escalado = np.ones((5,5), dtype=np.uint8)
        self.umbral_dist_escalado = 50
        self.umbral_fusion_escalado = 40
        self.area_moto_max_escalada = 1000
        self.area_coche_max_escalada = 10000
        self.font_peque = 0.5
        self.grosor_grande = 2
        self.grosor_peque = 1


    def setup_panel_carga(self):
        frame = ttk.LabelFrame(self.frame_controles, text="1. Controles Principales")
        frame.pack(fill=tk.X, pady=5)

        frame_load = ttk.Frame(frame)
        frame_load.pack(fill=tk.X)
        self.btn_load_video = ttk.Button(frame_load, text="Cargar Vídeo", command=self.load_video)
        self.btn_load_video.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.btn_load_bg = ttk.Button(frame_load, text="Cargar Fondo Estático", command=self.load_background)
        self.btn_load_bg.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5, pady=5)
        
        self.lbl_video_path = ttk.Label(frame, text="Vídeo no cargado", relief=tk.SUNKEN)
        self.lbl_video_path.pack(fill=tk.X, expand=True, padx=5, pady=2)
        self.lbl_bg_path = ttk.Label(frame, text="Fondo no cargado", relief=tk.SUNKEN)
        self.lbl_bg_path.pack(fill=tk.X, expand=True, padx=5, pady=2)
        
        self.btn_gen_bg = ttk.Button(frame, text="Generar Fondo desde Vídeo", command=self.generar_fondo)
        self.btn_gen_bg.pack(fill=tk.X, expand=True, padx=5, pady=5)

        # --- Botones de Cargar/Guardar Configuración ---
        frame_config = ttk.Frame(frame)
        frame_config.pack(fill=tk.X, pady=5)
        self.btn_load_config = ttk.Button(frame_config, text="Cargar Config (.json)", command=self.load_config)
        self.btn_load_config.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.btn_save_config = ttk.Button(frame_config, text="Guardar Config (.json)", command=self.save_config)
        self.btn_save_config.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)

        # --- Botones de Start/Stop ---
        frame_start = ttk.Frame(frame)
        frame_start.pack(fill=tk.X, pady=10)
        self.btn_start = ttk.Button(frame_start, text="INICIAR", command=self.start_processing)
        self.btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.btn_stop = ttk.Button(frame_start, text="DETENER", command=self.stop_processing, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)

    def setup_panel_tabs(self):
        frame = ttk.LabelFrame(self.frame_controles, text="2. Parámetros")
        frame.pack(fill=tk.BOTH, expand=True, pady=5)
        notebook = ttk.Notebook(frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Variables de Tkinter (con los defaults para trafico.mp4) ---
        self.params_vars["escala"] = tk.DoubleVar(value=0.5)
        self.params_vars["umbral_sensibilidad"] = tk.IntVar(value=30)
        self.params_vars["umbral_fusion_base"] = tk.IntVar(value=60)
        self.params_vars["min_area_base"] = tk.IntVar(value=250)
        self.params_vars["kernel_size_base"] = tk.IntVar(value=7)
        self.params_vars["umbral_dist_base"] = tk.IntVar(value=50)
        self.params_vars["max_frames_perdido"] = tk.IntVar(value=10)
        self.params_vars["frames_para_confirmar"] = tk.IntVar(value=5)
        self.params_vars["roi_base"] = tk.StringVar(value="280, 965, 0, 1920")
        self.params_vars["metodo_fondo"] = tk.StringVar(value="estatico")
        self.params_vars["frames_calentamiento"] = tk.IntVar(value=100)
        self.params_vars["orientacion_via"] = tk.StringVar(value="vertical")
        self.params_vars["factor_perspectiva_max"] = tk.DoubleVar(value=10.0)
        self.params_vars["filtro_sentido"] = tk.StringVar(value="None")
        self.params_vars["mostrar_texto_velocidad"] = tk.BooleanVar(value=True)
        self.params_vars["mostrar_texto_sentido"] = tk.BooleanVar(value=True)
        self.params_vars["mostrar_id"] = tk.BooleanVar(value=True)
        self.params_vars["mostrar_roi"] = tk.BooleanVar(value=True)
        self.params_vars["colorear_por"] = tk.StringVar(value="velocidad")
        self.params_vars["vel_min_color"] = tk.DoubleVar(value=4.0)
        self.params_vars["vel_max_color"] = tk.DoubleVar(value=20.0)
        self.params_vars["pixeles_por_metro"] = tk.DoubleVar(value=37.1)
        self.params_vars["mostrar_contadores"] = tk.BooleanVar(value=True)
        self.params_vars["mostrar_contador_activos"] = tk.BooleanVar(value=True)
        self.params_vars["mostrar_contador_historico"] = tk.BooleanVar(value=True)
        self.params_vars["mostrar_contador_subiendo"] = tk.BooleanVar(value=True)
        self.params_vars["mostrar_contador_bajando"] = tk.BooleanVar(value=True)
        self.params_vars["mostrar_tipo_coche"] = tk.BooleanVar(value=True)
        self.params_vars["area_moto_max_base"] = tk.IntVar(value=1000)
        self.params_vars["area_coche_max_base"] = tk.IntVar(value=10000)
        self.params_vars["retraso_sentido"] = tk.IntVar(value=15)
        self.params_vars["aspect_ratio_moto_max"] = tk.DoubleVar(value=1.0)
        self.params_vars["aspect_ratio_camion_min"] = tk.DoubleVar(value=1.1)
        
        # --- Pestaña 1: Calibración ---
        tab_cal = ttk.Frame(notebook)
        notebook.add(tab_cal, text='Calibración')
        self.crear_slider(tab_cal, "Escala de Procesamiento:", self.params_vars["escala"], 0.1, 1.0, 0.05)
        self.crear_radio(tab_cal, "Orientación Vía:", self.params_vars["orientacion_via"], ["vertical", "horizontal"])
        ttk.Label(tab_cal, text="ROI Base (y1, y2, x1, x2):").pack(anchor=tk.W)
        ttk.Entry(tab_cal, textvariable=self.params_vars["roi_base"]).pack(fill=tk.X, padx=5)
        self.crear_slider(tab_cal, "Factor Perspectiva (lejos):", self.params_vars["factor_perspectiva_max"], 1.0, 20.0, 0.1)
        self.crear_slider(tab_cal, "Píxeles / Metro (en zona 1.0):", self.params_vars["pixeles_por_metro"], 0.0, 200.0, 0.1)
        self.crear_slider(tab_cal, "Retraso Sentido (frames):", self.params_vars["retraso_sentido"], 1, 50, 1)

        # --- Pestaña 2: Detección ---
        tab_det = ttk.Frame(notebook)
        notebook.add(tab_det, text='Detección')
        self.crear_radio(tab_det, "Método de Fondo:", self.params_vars["metodo_fondo"], ["estatico", "dinamico"])
        self.crear_slider(tab_det, "Calentamiento MOG2 (frames):", self.params_vars["frames_calentamiento"], 0, 500, 10)
        self.crear_slider(tab_det, "Sensibilidad Detección:", self.params_vars["umbral_sensibilidad"], 1, 255, 1)
        self.crear_slider(tab_det, "Área Mínima Base:", self.params_vars["min_area_base"], 10, 5000, 10)
        self.crear_slider(tab_det, "Tamaño Kernel (impar):", self.params_vars["kernel_size_base"], 1, 11, 2) # Step 2 for odd
        self.crear_slider(tab_det, "Umbral Fusión Blobs:", self.params_vars["umbral_fusion_base"], 0, 200, 5)

        # --- Pestaña 3: Tracking ---
        tab_track = ttk.Frame(notebook)
        notebook.add(tab_track, text='Tracking')
        self.crear_slider(tab_track, "Distancia Asociación:", self.params_vars["umbral_dist_base"], 10, 500, 5)
        self.crear_slider(tab_track, "Paciencia Oclusión (frames):", self.params_vars["max_frames_perdido"], 1, 100, 1)
        self.crear_slider(tab_track, "Confirmación (frames):", self.params_vars["frames_para_confirmar"], 1, 50, 1)
        
        # --- Pestaña 4: Visualización ---
        tab_vis = ttk.Frame(notebook)
        notebook.add(tab_vis, text='Visualización')
        ttk.Checkbutton(tab_vis, text="Mostrar Contadores", variable=self.params_vars["mostrar_contadores"]).pack(anchor=tk.W)
        ttk.Checkbutton(tab_vis, text="Mostrar ID", variable=self.params_vars["mostrar_id"]).pack(anchor=tk.W)
        ttk.Checkbutton(tab_vis, text="Mostrar Velocidad", variable=self.params_vars["mostrar_texto_velocidad"]).pack(anchor=tk.W)
        ttk.Checkbutton(tab_vis, text="Mostrar Sentido", variable=self.params_vars["mostrar_texto_sentido"]).pack(anchor=tk.W)
        ttk.Checkbutton(tab_vis, text="Mostrar Tipo", variable=self.params_vars["mostrar_tipo_coche"]).pack(anchor=tk.W)
        ttk.Checkbutton(tab_vis, text="Mostrar ROI", variable=self.params_vars["mostrar_roi"]).pack(anchor=tk.W)
        self.crear_radio(tab_vis, "Colorear BBox por:", self.params_vars["colorear_por"], ["None", "sentido", "tipo", "velocidad"])
        self.crear_radio(tab_vis, "Filtrar por Sentido:", self.params_vars["filtro_sentido"], ["None", "SUBE", "BAJA"])
        self.crear_slider(tab_vis, "Vel. Mín. (Color):", self.params_vars["vel_min_color"], 0.0, 20.0, 0.5)
        self.crear_slider(tab_vis, "Vel. Máx. (Color):", self.params_vars["vel_max_color"], 5.0, 50.0, 0.5)

        # --- Pestaña 5: Clasificación ---
        tab_class = ttk.Frame(notebook)
        notebook.add(tab_class, text='Clasificación')
        self.crear_slider(tab_class, "Área Máx. Moto (Base):", self.params_vars["area_moto_max_base"], 100, 10000, 100)
        self.crear_slider(tab_class, "Área Máx. Coche (Base):", self.params_vars["area_coche_max_base"], 1000, 50000, 100)
        self.crear_slider(tab_class, "Aspect Ratio Máx. Moto (w/h):", self.params_vars["aspect_ratio_moto_max"], 0.1, 2.0, 0.05)
        self.crear_slider(tab_class, "Aspect Ratio Mín. Camión (w/h):", self.params_vars["aspect_ratio_camion_min"], 0.5, 3.0, 0.05)

    def crear_slider(self, parent, label, variable, from_, to, step=1.0):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(frame, text=label).pack(anchor=tk.W)
        slider_frame = ttk.Frame(frame)
        slider_frame.pack(fill=tk.X)
        is_double = isinstance(variable, tk.DoubleVar)
        
        # Etiqueta para el valor (se actualiza con el comando)
        value_label = ttk.Label(slider_frame, text=f"{variable.get():.2f}" if is_double else f"{variable.get()}", width=6, anchor=tk.E)
        value_label.pack(side=tk.RIGHT, padx=5)
        
        def on_slide(val_str):
            val = float(val_str)
            if is_double:
                value_label.config(text=f"{val:.2f}")
            else:
                value_label.config(text=f"{int(val)}")

        # 'resolution' es el argumento correcto para el 'step' en un Scale
        ttk.Scale(
            slider_frame, 
            from_=from_, 
            to=to, 
            variable=variable, 
            orient=tk.HORIZONTAL,
            command=on_slide,
            length=300,
            # Corregido: 'resolution' no es un param de ttk.Scale, 
            # pero el 'step' en la lógica del slider se maneja con 'command' y el 'step' del 'Scale' es implícito. 
        ).pack(fill=tk.X, expand=True)

    def crear_radio(self, parent, label, variable, values):
        ttk.Label(parent, text=label).pack(anchor=tk.W, padx=5, pady=5)
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X)
        for val in values:
            ttk.Radiobutton(frame, text=val.capitalize(), variable=variable, value=val).pack(side=tk.LEFT, padx=5)

    def setup_panel_video(self):
        frame_metricas = ttk.LabelFrame(self.frame_principal, text="Contadores en Tiempo Real")
        frame_metricas.pack(fill=tk.X, pady=5)
        
        self.metric_labels = {
            "activos": ttk.Label(frame_metricas, text="Activos: 0", font=("Arial", 14, "bold")),
            "historico": ttk.Label(frame_metricas, text="Total: 0", font=("Arial", 14, "bold")),
            "sentido1": ttk.Label(frame_metricas, text="Subiendo: 0", font=("Arial", 14, "bold"), foreground="green"),
            "sentido2": ttk.Label(frame_metricas, text="Bajando: 0", font=("Arial", 14, "bold"), foreground="red"),
            "motos": ttk.Label(frame_metricas, text="Motos: 0", font=("Arial", 12)),
            "coches": ttk.Label(frame_metricas, text="Coches: 0", font=("Arial", 12)),
            "camiones": ttk.Label(frame_metricas, text="Camiones: 0", font=("Arial", 12))
        }
        
        self.metric_labels["activos"].grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        self.metric_labels["historico"].grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
        self.metric_labels["sentido1"].grid(row=0, column=1, padx=20, pady=5, sticky=tk.W)
        self.metric_labels["sentido2"].grid(row=1, column=1, padx=20, pady=5, sticky=tk.W)
        self.metric_labels["motos"].grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)
        self.metric_labels["coches"].grid(row=2, column=1, padx=20, pady=5, sticky=tk.W)
        self.metric_labels["camiones"].grid(row=2, column=2, padx=20, pady=5, sticky=tk.W)
        frame_metricas.grid_columnconfigure(3, weight=1)

        frame_video = ttk.LabelFrame(self.frame_principal, text="Visor de Vídeo")
        frame_video.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.video_label = ttk.Label(frame_video, anchor=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True)

    def load_video(self):
        if self.processing_active:
            messagebox.showwarning("Aviso", "Detén el procesamiento actual antes de cargar un nuevo vídeo.")
            return
        path = filedialog.askopenfilename(filetypes=[("Archivos de vídeo", "*.mp4 *.avi *.mov"), ("Todos", "*.*")])
        if path:
            self.video_path = path
            self.lbl_video_path.config(text=os.path.basename(path))

    def load_background(self):
        if self.processing_active:
            messagebox.showwarning("Aviso", "Detén el procesamiento actual antes de cargar un nuevo fondo.")
            return
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.png"), ("Todos", "*.*")])
        if path:
            self.fondo_path = path
            self.lbl_bg_path.config(text=os.path.basename(path))

    # --- Funciones de configuración ---
    def load_config(self):
        """Carga los parámetros desde un fichero JSON."""
        if self.processing_active:
            messagebox.showwarning("Aviso", "Detén el procesamiento actual para cargar una configuración.")
            return
            
        path = filedialog.askopenfilename(filetypes=[("Archivos JSON", "*.json"), ("Todos", "*.*")])
        if not path:
            return
            
        try:
            with open(path, 'r') as f:
                config_data = json.load(f)
            
            # Recorremos los datos cargados y los 'seteamos' en nuestras variables de Tkinter
            for key, value in config_data.items():
                if key in self.params_vars:
                    self.params_vars[key].set(value)
            
            messagebox.showinfo("Éxito", "Configuración cargada correctamente.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el fichero JSON: {e}")

    def save_config(self):
        """Guarda los parámetros actuales en un fichero JSON."""
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("Archivos JSON", "*.json"), ("Todos", "*.*")],
            title="Guardar configuración como..."
        )
        if not path:
            return
            
        try:
            # Creamos un diccionario 'limpio' con los valores actuales
            current_config = {key: var.get() for key, var in self.params_vars.items()}
            
            with open(path, 'w') as f:
                json.dump(current_config, f, indent=4) # indent=4 es para que el JSON sea legible
            
            messagebox.showinfo("Éxito", f"Configuración guardada en: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar el fichero JSON: {e}")

    def generar_fondo(self):
        """Llama a 'obtener_fondo' con feedback visual y sin bloquear"""
        if not self.video_path:
            messagebox.showerror("Error", "Carga un vídeo primero para poder generarle un fondo.")
            return
        
        # --- Evitar que se ejecute si el vídeo está corriendo ---
        if self.processing_active:
            messagebox.showwarning("Aviso", "Detén el procesamiento actual antes de generar un fondo.")
            return

        # 1. Crear el popup de "Cargando"
        popup = tk.Toplevel(self.root)
        popup.title("Procesando")
        popup.update_idletasks()  # Asegura que se calcule correctamente el tamaño
        ancho, alto = 300, 100
        x = (popup.winfo_screenwidth() // 2) - (ancho // 2)
        y = (popup.winfo_screenheight() // 2) - (alto // 2)

        popup.geometry(f"{ancho}x{alto}+{x}+{y}")   
        popup.geometry("300x100")
        popup.transient(self.root)
        popup.grab_set() 
        ttk.Label(popup, text="Generando fondo, por favor espere...").pack(padx=20, pady=20)
        
        # 2. Forzar a Tkinter a dibujar el popup
        popup.update() 

        try:
            # 3. Ejecutar la tarea pesada
            obtener_fondo(self.video_path) # Esta es tu función de functions.py
            
            # 4. Cerrar el popup y mostrar éxito
            popup.destroy()
            messagebox.showinfo("Éxito", "¡Fondo generado y guardado en la carpeta 'images'!")
        except Exception as e:
            # 5. Manejar errores
            popup.destroy()
            messagebox.showerror("Error", f"No se pudo generar el fondo: {e}")

    def start_processing(self):
        if not self.video_path:
            messagebox.showerror("Error", "¡Por favor, carga un vídeo primero!")
            return
            
        # --- Resetear el contador histórico ---
        Vehiculo._next_id = 0
            
        self.params = {key: var.get() for key, var in self.params_vars.items()}
        
        try:
            self.params["roi_base"] = [int(x.strip()) for x in self.params["roi_base"].split(',')]
            if len(self.params["roi_base"]) != 4: raise ValueError
        except Exception:
            messagebox.showerror("Error", "Formato de ROI incorrecto. Debe ser: y1, y2, x1, x2")
            return
            
        if self.params["filtro_sentido"] == "None": self.params["filtro_sentido"] = None
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"No se pudo abrir el vídeo: {self.video_path}")
            return
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0: self.fps = 30.0 
        self.delay = int(1000 / self.fps) 
        
        self.frame_num = 0
        
        # --- Definir new_size ANTES de usarlo ---
        escala = self.params["escala"]
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * escala)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * escala)
        self.new_size = (w, h)
        if w == 0 or h == 0:
            messagebox.showerror("Error", "No se pudieron leer las dimensiones del vídeo.")
            return

        # 4. Pre-calcular parámetros escalados
        self.min_area_escalada = self.params["min_area_base"] * (escala**2)
        ksv = int(np.ceil(self.params["kernel_size_base"] * escala)) // 2 * 2 + 1
        if ksv < 1: ksv = 1 
        self.kernel_escalado = cv2.getStructuringElement(cv2.MORPH_RECT, (ksv, ksv))
        self.umbral_dist_escalado = self.params["umbral_dist_base"] * escala
        self.umbral_fusion_escalado = self.params["umbral_fusion_base"] * escala
        self.area_moto_max_escalada = self.params["area_moto_max_base"] * (escala**2)
        self.area_coche_max_escalada = self.params["area_coche_max_base"] * (escala**2)
        self.font_peque = max(0.4, 0.8 * escala)
        self.grosor_grande = max(1, int(2 * escala))
        self.grosor_peque = max(1, int(1 * escala))
        
        self.gestor = GestorVehiculos(
            umbral_distancia=self.umbral_dist_escalado,
            max_frames_perdido=self.params["max_frames_perdido"]
        )
        
        if self.params["metodo_fondo"] == 'estatico':
            if not self.fondo_path:
                messagebox.showerror("Error", "Método estático seleccionado, pero no se ha cargado un fondo.")
                return
            
            try:
                # --- Usamos el método de numpy para leer paths con 'º' ---
                with open(self.fondo_path, "rb") as f:
                    bytes_leidos = f.read()
                numpy_array = np.asarray(bytearray(bytes_leidos), dtype=np.uint8)
                fondo_cv = cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)
                if fondo_cv is None: raise Exception("la imagen de fondo está corrupta.")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo leer la imagen de fondo: {e}")
                return
            
            self.fondo_redimensionado = cv2.resize(fondo_cv, self.new_size).astype(np.uint8)
        else:
            self.sustractor_fondo = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=self.params["umbral_sensibilidad"], detectShadows=True)
            
        self.mask_roi = np.ones((self.new_size[1], self.new_size[0]), dtype=np.uint8) * 255
        self.roi_escalada = None
        if self.params["roi_base"]:
            self.roi_escalada = [int(x * escala) for x in self.params["roi_base"]]
            self.mask_roi = np.zeros((self.new_size[1], self.new_size[0]), dtype=np.uint8)
            self.mask_roi[self.roi_escalada[0]:self.roi_escalada[1], self.roi_escalada[2]:self.roi_escalada[3]] = 255

        self.processing_active = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        # --- Desactivar botones ---
        self.btn_gen_bg.config(state=tk.DISABLED)
        self.btn_load_video.config(state=tk.DISABLED)
        self.btn_load_bg.config(state=tk.DISABLED)
        self.btn_load_config.config(state=tk.DISABLED)
        self.btn_save_config.config(state=tk.DISABLED)
        
        self.video_loop()

    def stop_processing(self):
        self.processing_active = False
        if self.cap:
            self.cap.release()
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        # --- Reactivar botones ---
        self.btn_gen_bg.config(state=tk.NORMAL)
        self.btn_load_video.config(state=tk.NORMAL)
        self.btn_load_bg.config(state=tk.NORMAL)
        self.btn_load_config.config(state=tk.NORMAL)
        self.btn_save_config.config(state=tk.NORMAL)
        print("Procesamiento detenido.")

    def video_loop(self):
        """El corazón de la app: procesa un frame y se auto-llama."""
        
        if not self.processing_active:
            return

        try:
            ret, frame = self.cap.read()
            if not ret:
                self.stop_processing()
                messagebox.showinfo("Info", "Fin del vídeo.")
                return
                
            self.frame_num += 1
            frame = cv2.resize(frame, self.new_size)

            # --- Calentamiento MOG2 ---
            if self.params["metodo_fondo"] == 'dinamico' and self.frame_num < self.params["frames_calentamiento"]:
                self.sustractor_fondo.apply(frame)
                detecciones_limpias = []
            else:
                # --- Detección ---
                if self.params["metodo_fondo"] == 'estatico':
                    diff = cv2.absdiff(frame, self.fondo_redimensionado)
                    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    _, fgmask = cv2.threshold(diff, self.params["umbral_sensibilidad"], 255, cv2.THRESH_BINARY)
                else: 
                    fgmask_con_sombras = self.sustractor_fondo.apply(frame)
                    _, fgmask = cv2.threshold(fgmask_con_sombras, 250, 255, cv2.THRESH_BINARY)
                
                fgmask = cv2.bitwise_and(fgmask, self.mask_roi)
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, self.kernel_escalado)
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel_escalado)
                
                contornos, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                detecciones_sucias = []
                for c in contornos:
                    if cv2.contourArea(c) < self.min_area_escalada:
                        continue
                    x, y, w, h = cv2.boundingRect(c)
                    detecciones_sucias.append((x, y, w, h))
                
                detecciones_limpias = fusionar_detecciones_cercanas(detecciones_sucias, self.umbral_fusion_escalado)
            
            self.gestor.actualizar(detecciones_limpias, frame, self.frame_num)

            # --- Dibujar resultados ---
            contadores_frame = { "activos": 0, "historico": Vehiculo._next_id, "sentido1": 0, "sentido2": 0, "motos": 0, "coches": 0, "camiones": 0 }
            
            if self.params["orientacion_via"] == 'vertical':
                tag_sentido_1 = 'SUBE'; tag_sentido_2 = 'BAJA'
            else:
                tag_sentido_1 = 'IZQUIERDA'; tag_sentido_2 = 'DERECHA'

            for v in self.gestor.vehiculos_activos():
                
                if v.frames_activo <= self.params["frames_para_confirmar"]:
                    continue 

                if v.sentido is None and v.frames_activo > self.params["retraso_sentido"]:
                    vx, vy = v.velocidad
                    if self.params["orientacion_via"] == 'vertical':
                        if vy < -1.0: v.sentido = tag_sentido_1
                        elif vy > 1.0: v.sentido = tag_sentido_2
                    else:
                        if vx < -1.0: v.sentido = tag_sentido_1
                        elif vx > 1.0: v.sentido = tag_sentido_2
                
                if self.params["filtro_sentido"] and v.sentido != self.params["filtro_sentido"]:
                    continue 

                contadores_frame["activos"] += 1
                if v.sentido == tag_sentido_1: contadores_frame["sentido1"] += 1
                elif v.sentido == tag_sentido_2: contadores_frame["sentido2"] += 1

                if v.tipo == 'Indefinido':
                    w, h = v.bbox[2], v.bbox[3]
                    area_bbox = w * h
                    aspect_ratio = w / (h + 1e-6)
                    if area_bbox < self.area_moto_max_escalada and aspect_ratio < self.params["aspect_ratio_moto_max"]:
                        v.tipo = 'Moto'
                    elif area_bbox > self.area_coche_max_escalada and aspect_ratio > self.params["aspect_ratio_camion_min"]:
                        v.tipo = 'Camion'
                    else:
                        v.tipo = 'Coche'

                if v.tipo == 'Moto': contadores_frame["motos"] += 1
                elif v.tipo == 'Coche': contadores_frame["coches"] += 1
                elif v.tipo == 'Camion': contadores_frame["camiones"] += 1

                x, y, w, h = map(int, v.bbox)
                velocidad_mag_bruta = np.linalg.norm(v.velocidad)
                factor_correccion = 1.0
                if self.params["roi_base"] and self.params["orientacion_via"] == 'vertical':
                    y1_roi = self.roi_escalada[0]; y2_roi = self.roi_escalada[1]
                    y_coche = v.centroide[1]
                    y_relativa = np.clip((y_coche - y1_roi) / (y2_roi - y1_roi + 1e-6), 0.0, 1.0)
                    factor_correccion = self.params["factor_perspectiva_max"] - (y_relativa * (self.params["factor_perspectiva_max"] - 1.0))
                velocidad_corregida = velocidad_mag_bruta * factor_correccion

                color_caja = (0, 255, 0)
                cp = self.params["colorear_por"]
                if cp == 'sentido':
                    if v.sentido == tag_sentido_1: color_caja = (0, 255, 0)
                    elif v.sentido == tag_sentido_2: color_caja = (0, 0, 255)
                    else: color_caja = (255, 0, 0)
                elif cp == 'tipo':
                    if v.tipo == 'Moto': color_caja = (255, 0, 255)
                    elif v.tipo == 'Camion': color_caja = (255, 255, 0)
                    else: color_caja = (0, 255, 0)
                elif cp == 'velocidad':
                    vel_norm = np.clip((velocidad_corregida - self.params["vel_min_color"]) / (self.params["vel_max_color"] - self.params["vel_min_color"]), 0.0, 1.0)
                    R = int(vel_norm * 255); B = int((1 - vel_norm) * 255)
                    color_caja = (B, 0, R)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color_caja, self.grosor_grande)
                
                y_offset_texto = y + h + int(15 * self.params["escala"])
                
                if self.params["mostrar_id"]:
                    cv2.putText(frame, f"ID {v.id}", (x, y - int(10 * self.params["escala"])), 
                                cv2.FONT_HERSHEY_SIMPLEX, self.font_peque, (255,255,0), self.grosor_peque)
                
                if self.params["mostrar_texto_velocidad"]:
                    if self.params["pixeles_por_metro"] > 0 and self.fps > 0:
                        velocidad_ms = (velocidad_corregida * self.fps) / self.params["pixeles_por_metro"]
                        velocidad_kmh = velocidad_ms * 3.6
                        texto_vel = f"{velocidad_kmh:.1f} km/h"
                    else:
                        unidad = " u/f" if (self.params["roi_base"] and self.params["orientacion_via"] == 'vertical') else " p/f"
                        texto_vel = f"{velocidad_corregida:.1f}{unidad}"
                    cv2.putText(frame, texto_vel, (x, y_offset_texto), 
                                cv2.FONT_HERSHEY_SIMPLEX, self.font_peque, (0, 255, 255), self.grosor_peque)
                    y_offset_texto += int(15 * self.params["escala"])
                
                if self.params["mostrar_texto_sentido"]:
                    color_sentido = (255, 0, 0); texto_sentido = '(...)'
                    if v.sentido == tag_sentido_1: color_sentido = (0, 255, 0); texto_sentido = tag_sentido_1
                    elif v.sentido == tag_sentido_2: color_sentido = (0, 0, 255); texto_sentido = tag_sentido_2
                    cv2.putText(frame, texto_sentido, (x, y_offset_texto), 
                                cv2.FONT_HERSHEY_SIMPLEX, self.font_peque, color_sentido, self.grosor_peque)
                    y_offset_texto += int(15 * self.params["escala"])
                
                if self.params["mostrar_tipo_coche"]:
                    cv2.putText(frame, v.tipo, (x, y_offset_texto),
                                cv2.FONT_HERSHEY_SIMPLEX, self.font_peque, (255, 0, 255), self.grosor_peque)
            
            if self.params["mostrar_roi"] and self.roi_escalada is not None:
                 cv2.rectangle(frame, (self.roi_escalada[2], self.roi_escalada[0]), (self.roi_escalada[3], self.roi_escalada[1]), (255, 0, 0), self.grosor_grande)

            # --- Actualizar Métricas (Contadores) ---
            if self.params["mostrar_contadores"]:
                # (Actualizamos los labels de sentido por si cambia la orientación)
                self.metric_labels["sentido1"].config(text=f"{tag_sentido_1}: {contadores_frame['sentido1']}")
                self.metric_labels["sentido2"].config(text=f"{tag_sentido_2}: {contadores_frame['sentido2']}")
                
                self.metric_labels["activos"].config(text=f"Activos: {contadores_frame['activos']}")
                self.metric_labels["historico"].config(text=f"Total: {contadores_frame['historico']}")
                self.metric_labels["motos"].config(text=f"Motos: {contadores_frame['motos']}")
                self.metric_labels["coches"].config(text=f"Coches: {contadores_frame['coches']}")
                self.metric_labels["camiones"].config(text=f"Camiones: {contadores_frame['camiones']}")

            # --- Mostrar el frame en Tkinter ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            label_w = self.video_label.winfo_width()
            label_h = self.video_label.winfo_height()
            
            if label_w > 1 and label_h > 1:
                img_ratio = img.width / img.height
                label_ratio = label_w / label_h
                
                if label_ratio > img_ratio: 
                    new_h = label_h
                    new_w = int(img_ratio * label_h)
                else: 
                    new_w = label_w
                    new_h = int(label_w / img_ratio)
                
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        except Exception as e:
            print(f"Error en video_loop: {e}")
            self.stop_processing()
            messagebox.showerror("Error de Procesamiento", f"Ha ocurrido un error: {e}")
            return

        # --- Programar el siguiente frame ---
        self.root.after(self.delay, self.video_loop)

# --- Punto de entrada de la aplicación ---
if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use('vista') 
    except tk.TclError:
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass 
            
    app = RadarApp(root)
    root.mainloop()