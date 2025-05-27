import pandas as pd
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
from tkinter.ttk import Notebook
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DataApp(tb.Window):
    def __init__(self):
        super().__init__(title="DataApp con ttkbootstrap", themename="flatly")
        self.geometry("1200x700")
        self.state('zoomed')
        self.df = None

        self.init_ui()

    def init_ui(self):
        # Crear el contenedor principal con pestañas
        self.notebook = Notebook(self)
        self.notebook.pack(fill=BOTH, expand=YES)

        # Pestaña principal para tabla + gráficos
        self.principal_tab = tb.Frame(self.notebook)
        self.notebook.add(self.principal_tab, text='Vista Principal')

        # Frame superior con botones
        top_frame = tb.Frame(self.principal_tab)
        top_frame.pack(side=TOP, fill=X, pady=5)

        tb.Button(top_frame, text="Cargar CSV", command=self.cargar_csv, bootstyle="primary").pack(side=LEFT, padx=5)
        tb.Button(top_frame, text="Estandarizar", command=self.estandarizar_datos, bootstyle="info").pack(side=LEFT, padx=5)
        tb.Button(top_frame, text="Sobremuestrear", command=self.sobremuestrear_datos, bootstyle="warning").pack(side=LEFT, padx=5)
        tb.Button(top_frame, text="Mostrar Gráficos", command=self.mostrar_graficos, bootstyle="success").pack(side=LEFT, padx=5)

        # Área dividida en 2: tabla (izquierda) y gráficos (derecha)
        self.paned = tb.PanedWindow(self.principal_tab, orient=HORIZONTAL)
        self.paned.pack(fill=BOTH, expand=YES)

        # Panel izquierdo: tabla
        self.tabla_frame = tb.Frame(self.paned)
        self.paned.add(self.tabla_frame, weight=2)

        self.tree_frame = tb.Frame(self.tabla_frame)
        self.tree_frame.pack(fill=BOTH, expand=YES)

        self.tree = tb.Treeview(self.tree_frame, show='headings')
        self.tree.pack(side=LEFT, fill=BOTH, expand=YES)

        vsb = tb.Scrollbar(self.tree_frame, orient=VERTICAL, command=self.tree.yview)
        vsb.pack(side=RIGHT, fill=Y)
        self.tree.configure(yscrollcommand=vsb.set)

        hsb = tb.Scrollbar(self.tabla_frame, orient=HORIZONTAL, command=self.tree.xview)
        hsb.pack(fill=X)
        self.tree.configure(xscrollcommand=hsb.set)

        # Panel derecho: zona para gráficos
        self.grafico_frame = tb.Frame(self.paned)
        self.paned.add(self.grafico_frame, weight=1)


    def cargar_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path:
            try:
                self.df = pd.read_csv(path)
                self.mostrar_dataframe(self.df)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el archivo: {e}")

    def mostrar_dataframe(self, df):
        self.tree.delete(*self.tree.get_children())
        columnas = ['ID'] + list(df.columns)
        self.tree['columns'] = columnas

        for i, col in enumerate(columnas):
            self.tree.heading(col, text=col)
            if col == 'ID':
                max_len = max(len(str(idx)) for idx in df.index)
                self.tree.column(col, width=max_len * 30, stretch=False)
                continue
            else:
                max_len = max(len(str(v)) for v in df[col].astype(str))
                max_len = max(len(str(col)), max_len)
            self.tree.column(col, width=max_len * 10, stretch=False)


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

    def estandarizar_datos(self):
        if self.df is not None:
            try:
                numeric_cols = self.df.select_dtypes(include='number').columns
                self.df[numeric_cols] = StandardScaler().fit_transform(self.df[numeric_cols])
                self.mostrar_dataframe(self.df)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo estandarizar: {e}")

    def sobremuestrear_datos(self):
        if self.df is not None:
            try:
                target = self.df.columns[-1]
                X = self.df.iloc[:, :-1]
                y = self.df.iloc[:, -1]
                X_resampled, y_resampled = SMOTE().fit_resample(X, y)
                self.df = pd.DataFrame(X_resampled, columns=X.columns)
                self.df[target] = y_resampled
                self.mostrar_dataframe(self.df)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo sobremuestrear: {e}")

    def mostrar_graficos(self):
        for widget in self.grafico_frame.winfo_children():
            widget.destroy()

        if self.df is None:
            return

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        fig.tight_layout(pad=3.0)

        num_cols = self.df.select_dtypes(include='number').columns
        if len(num_cols) > 0:
            self.df[num_cols[0]].hist(ax=axs[0])
            axs[0].set_title(f'Histograma: {num_cols[0]}')

        if len(num_cols) > 1:
            axs[1].scatter(self.df[num_cols[0]], self.df[num_cols[1]], alpha=0.5)
            axs[1].set_title(f'Dispersión: {num_cols[0]} vs {num_cols[1]}')

        cat_cols = self.df.select_dtypes(include='object').columns
        if len(cat_cols) > 0:
            self.df[cat_cols[0]].value_counts().plot(kind='bar', ax=axs[2])
            axs[2].set_title(f'Frecuencias: {cat_cols[0]}')

        canvas = FigureCanvasTkAgg(fig, master=self.grafico_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=YES)


if __name__ == "__main__":
    app = DataApp()
    app.mainloop()
