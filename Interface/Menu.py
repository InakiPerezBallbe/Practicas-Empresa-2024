import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scripts.Preprocessing import readDataframe, standarize, oversample

class DataApp(ttk.Window):
    def __init__(self): 
        super().__init__(title="Ánalisis de Datos")
        self.iconbitmap('./images/Icono.ico')
        self.geometry("1200x700")
        self.state('zoomed')
        self.df = None

        # Crear el contenedor prdincipal con pestañas
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=BOTH, expand=YES)

        # Pestaña principal para tabla + gráficos
        self.preprocess_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.preprocess_tab, text='Preprocesamiento')

        # Área dividida en 2: tabla (izquierda) y gráficos (derecha)
        self.pw = ttk.PanedWindow(self.preprocess_tab, orient=ttk.HORIZONTAL)
        self.pw.pack(fill=BOTH, expand=YES)

        # Panel izquierdo: tabla
        self.left_frame = ttk.Frame(self.pw)
        self.left_frame.pack_propagate(False)
        self.left_frame.pack(expand=True, fill=BOTH) 
        self.pw.add(self.left_frame, weight=3)

        self.mostrar_cuadro_carga_csv(self.left_frame)

        # Panel derecho: zona para gráficos
        self.right_frame = ttk.Frame(self.pw)
        self.right_frame.pack_propagate(False) 
        self.pw.add(self.right_frame, weight=2)

        # Parte superior derecha con botones
        self.paned_right = ttk.PanedWindow(self.right_frame, orient=VERTICAL)
        self.paned_right.pack(fill=BOTH, expand=YES)

        self.topright_frame = ttk.Frame(self.paned_right)
        self.paned_right.add(self.topright_frame, weight=1)

        ttk.Button(self.topright_frame, text="Estandarizar", command=self.estandarizar_datos, bootstyle="info").pack(side=TOP, padx=5)
        ttk.Button(self.topright_frame, text="Sobremuestrear", command=self.sobremuestrear_datos, bootstyle="warning").pack(side=TOP, padx=5)

        self.bottomright_frame = ttk.Frame(self.paned_right)
        self.bottomright_frame.pack(fill=BOTH, expand=YES)
        self.paned_right.add(self.bottomright_frame, weight=3)

        self.menu_bar = ttk.Menu(self)
        self.config(menu=self.menu_bar)

    def mostrar_cuadro_carga_csv(self, panel):
        panel = ttk.Frame(panel, padding=20, relief="ridge", borderwidth=2)
        panel.place(relx=0.5, rely=0.5, anchor="center")

        ttk.Label(panel, text="Selecciona el archivo CSV:").pack(pady=5)
        ruta_entry = ttk.Entry(panel, width=50)
        ruta_entry.pack(pady=5)

        def buscar_archivo():
            ruta = filedialog.askopenfilename(filetypes=[("Archivos CSV", "*.csv")])
            ruta_entry.delete(0, END)
            ruta_entry.insert(0, ruta)

        ttk.Button(panel, text="Buscar", command=buscar_archivo, bootstyle="secondary").pack(pady=5)

        ttk.Label(panel, text="Separador:").pack(pady=5)
        separadores = {"Coma (,)": ",", "Punto y coma (;)": ";", "Tabulación (\t)": "\t", "Espacio": " "}
        separador_combo = ttk.Combobox(panel, values=list(separadores.keys()), state="readonly")
        separador_combo.set("Coma (,)")
        separador_combo.pack(pady=5)

        ttk.Label(panel, text="Codificación:").pack(pady=5)
        codificaciones = ["utf-8", "latin1", "ISO-8859-1"]
        codificacion_combo = ttk.Combobox(panel, values=codificaciones, state="readonly")
        codificacion_combo.set("utf-8")
        codificacion_combo.pack(pady=5)

        self.selected_column = None

        def cargar():
            ruta = ruta_entry.get()
            sep = separadores[separador_combo.get()]
            cod = codificacion_combo.get()
            panel.destroy()
            self.df = readDataframe(ruta, cod, sep)
            self.mostrar_dataframe(self.df)
            self.creargrafico()

        ttk.Button(panel, text="Cargar", command=cargar, bootstyle="success").pack(pady=10)

    def mostrar_dataframe(self, df):
        columnas = ['ID'] + list(df.columns)
        self.tree = ttk.Treeview(self.left_frame, columns=columnas, show='headings')

        for col in columnas:
            self.tree.heading(col, text=col)

            if col == 'ID':
                max_len = max(len(str(idx)) for idx in df.index)
                width = max(max_len * 8, 100)
            else:
                max_len = max(len(str(val)) for val in df[col])
                width = max(len(col) * 8, max_len * 8, 120)

            self.tree.column(col, width=width, anchor='w', stretch=False)

        # Insertar datos con colores
        color_columnas = ["#ccffcc", "#b3ffb3"]
        color_filas = ["#ffffff", "#f2f2f2"]

        for idx, row in df.iterrows():
            values = [idx] + list(row)
            fila_color = color_filas[idx % 2]
            tags = []
            for i in range(len(values)):
                col_color = color_columnas[i % 2]
                final_color = col_color if fila_color == "#ffffff" else "#e6ffe6"
                tag_name = f"tag_{idx}_{i}"
                self.tree.tag_configure(tag_name, background=final_color)
                tags.append(tag_name)
            self.tree.insert("", "end", iid=str(idx), values=values, tags=tags)

        vsb = ttk.Scrollbar(self.left_frame, orient='vertical', command=self.tree.yview, bootstyle="round")
        hsb = ttk.Scrollbar(self.left_frame, orient='horizontal', command=self.tree.xview, bootstyle="round")
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.left_frame.grid_rowconfigure(0, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)

        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
    def estandarizar_datos(self):
        if self.df is not None:
            try:
                numeric_cols = self.df.select_dtypes(include='number').columns
                self.df[numeric_cols] = StandardScaler().fit_transform(self.df[numeric_cols])
                self.mostrar_dataframe(self.df)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo estandarizar: {e}")

    def sobremuestrear_datos(self):
        if self.df is None or self.df.empty:
            messagebox.showwarning("Advertencia", "No hay datos cargados.")
            return

        dialog = ttk.Toplevel(self)
        dialog.title("Seleccionar columna para sobremuestreo")
        dialog.geometry("400x180")
        dialog.resizable(False, False)

        label = ttk.Label(dialog, text="Selecciona la columna objetivo (clase):")
        label.pack(pady=(20, 10), padx=20)

        column_var = ttk.StringVar()
        combo = ttk.Combobox(
            dialog,
            textvariable=column_var,
            values=list(self.df.columns),
            state="readonly",
            bootstyle="info"
        )
        combo.pack(padx=20, fill=X)
        combo.current(len(self.df.columns) - 1)

        def aplicar_smote():
            target_col = column_var.get()
            if not target_col:
                messagebox.showwarning("Advertencia", "Debe seleccionar una columna.")
                return

            try:
                self.df = oversample(target_col)
                self.mostrar_dataframe(self.df)
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo sobremuestrear: {e}")

        apply_btn = ttk.Button(dialog, text="Aplicar", bootstyle="success", command=aplicar_smote)
        apply_btn.pack(pady=20)
        dialog.grab_set()

    def show_message_on_plot(self, message):
        if self.ax and self.canvas:
            self.ax.clear()
            self.ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=10, color='red', wrap=True)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw()

    def plot_data(self):
        column_name = self.selected_column.get()
        plot_type = self.selected_plot_type.get()

        if not column_name:
            self.show_message_on_plot("Por favor, selecciona una columna.")
            return

        if self.ax is None or self.canvas is None or self.df is None:
            print("Error: Componentes del gráfico o DataFrame no inicializados.")
            return
            
        self.ax.clear() # Limpiar el eje anterior

        try:
            data_series = self.df[column_name]

            if plot_type == "Histograma":
                if pd.api.types.is_numeric_dtype(data_series):
                    data_series.plot(kind='hist', ax=self.ax, bins=15, color=ttk.Style().colors.primary, edgecolor='black')
                    self.ax.set_title(f"Histograma de '{column_name}'", fontsize=12)
                    self.ax.set_xlabel(column_name, fontsize=10)
                else:
                    self.show_message_on_plot(f"Histograma solo para columnas numéricas.\n'{column_name}' no es numérica.")
                    return

            elif plot_type == "Diagrama de Barras":
                if pd.api.types.is_categorical_dtype(data_series) or \
                pd.api.types.is_object_dtype(data_series) or \
                data_series.nunique() < 20: # Umbral para considerar numérico como discreto
                    counts = data_series.value_counts().sort_index()
                    counts.plot(kind='bar', ax=self.ax, color=ttk.Style().colors.info, edgecolor='black')
                    self.ax.set_title(f"Diagrama de Barras de '{column_name}'", fontsize=12)
                    self.ax.set_xlabel(column_name, fontsize=10)
                    plt.xticks(rotation=45, ha="right", fontsize=8)
                else:
                    self.show_message_on_plot(f"Diagrama de barras es mejor para datos categóricos\no numéricos discretos (pocos valores únicos).\nPrueba con un histograma para '{column_name}'.")
                    return
            
            self.ax.grid(axis='y', linestyle='--', alpha=0.7)

        except Exception as e:
            self.show_message_on_plot(f"Error al graficar: {e}", self.ax, self.canvas)
            return

        self.canvas.draw() # Redibujar el lienzo

    def creargrafico(self):
        self.selected_column = ttk.StringVar()
        self.selected_plot_type = ttk.StringVar(value="Histograma")

        self.selector = ttk.Frame(self.bottomright_frame)
        self.selector.pack(side=TOP, fill=X, pady=5)

        col_label = ttk.Label(self.selector, text="Seleccionar Columna:")
        col_label.pack(anchor=W)
        column_combobox = ttk.Combobox(
            self.selector,
            textvariable=self.selected_column,
            values=list(self.df.columns),
            state="readonly",
            bootstyle="info"
        )
        column_combobox.pack(fill=X, pady=(0, 15))
        column_combobox.bind("<<ComboboxSelected>>", lambda e: self.plot_data())
        if self.df.columns.any():
            column_combobox.current(0)

        plot_type_label = ttk.Label(self.selector, text="Tipo de Gráfico:")
        plot_type_label.pack(anchor=W)
        
        hist_radio = ttk.Radiobutton(
            self.selector,
            text="Histograma",
            variable=self.selected_plot_type,
            value="Histograma",
            command=self.plot_data,
            bootstyle="info-toolbutton"
        )
        hist_radio.pack(fill=X, pady=2)

        bar_radio = ttk.Radiobutton(
            self.selector,
            text="Diagrama de Barras",
            variable=self.selected_plot_type,
            value="Diagrama de Barras",
            command=self.plot_data,
            bootstyle="info-toolbutton"
        )
        bar_radio.pack(fill=X, pady=2)

        self.canvas_frame = ttk.Frame(self.bottomright_frame)
        self.canvas_frame.pack(side=BOTTOM, fill=BOTH, expand=YES)

        fig = Figure(figsize=(6, 4))
        self.ax = fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(side=TOP, fill=BOTH, expand=TRUE)

if __name__ == "__main__":
    app = DataApp()
    app.mainloop()
