import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
import dice_ml
from dice_ml.utils import helpers
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar datos
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
data = pd.read_csv("C:/Users/Usuario/OneDrive/Desktop/Practicas-Empresa-2024/Practicas-Empresa-2024/student-mat.csv", sep=";")

# Selección de características relevantes
features = ["studytime", "failures", "absences", "G1", "G2"]
target = "G3"

data["aprobado"] = (data["G3"] >= 10).astype(int)
X = data[features]
y = data["aprobado"]

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado de datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Crear objeto de modelo DiCE
dice_model = dice_ml.Model(model=model, backend="sklearn")
d = dice_ml.Data(dataframe=pd.concat([X, y], axis=1), continuous_features=features, outcome_name="aprobado")
exp = dice_ml.Dice(d, dice_model, method="random")

# Función para predecir
def predecir():
    try:
        # Obtener valores y validarlos
        horas = float(entry_horas.get())
        failures = int(entry_failures.get())
        absences = int(entry_absences.get())
        G1 = float(entry_G1.get())
        G2 = float(entry_G2.get())

        # Convertir valores a categorías para el modelo
        if horas < 2:
            studytime = 1
        elif 2 <= horas < 5:
            studytime = 2
        elif 5 <= horas < 10:
            studytime = 3
        else:
            studytime = 4

        failures = failures if failures < 3 else 4

        # Crear DataFrame con los datos ingresados
        new_student = pd.DataFrame([[studytime, failures, absences, G1, G2]], columns=features)

        # Hacer la predicción
        pred_clase = model.predict(new_student)[0]
        pred_proba = model.predict_proba(new_student)[0][pred_clase] * 100

        resultado = "Aprobado ✅" if pred_clase == 1 else "Suspendido ❌"
        messagebox.showinfo("Resultado", f"El estudiante ha **{resultado}** con una probabilidad del {round(pred_proba, 2)}%.")

    except ValueError:
        messagebox.showerror("Error", "Por favor, ingresa valores válidos.")


# Mapas para convertir valores numéricos en texto más comprensible
studytime_map = {
    1: "Menos de 2 horas",
    2: "Entre 2 y 5 horas",
    3: "Entre 5 y 10 horas",
    4: "Más de 10 horas"
}

failures_map = {
    0: "Ninguna",
    1: "1",
    2: "2",
    3: "3",
    4: "Más de 3"
}
# Función para generar contraejemplos
def generar_contrafactuales():
    try:
        # Obtener valores
        horas = float(entry_horas.get())
        failures = int(entry_failures.get())
        absences = int(entry_absences.get())
        G1 = float(entry_G1.get())
        G2 = float(entry_G2.get())

        # Convertir valores a categorías para el modelo
        if horas < 2:
            studytime = 1
        elif 2 <= horas < 5:
            studytime = 2
        elif 5 <= horas < 10:
            studytime = 3
        else:
            studytime = 4

        failures = failures if failures < 3 else 4

        # Crear DataFrame con los datos ingresados
        new_student = pd.DataFrame([[studytime, failures, absences, G1, G2]], columns=features)

        # Generar contraejemplos
        cf = exp.generate_counterfactuals(new_student, total_CFs=3, desired_class="opposite")

        # Limpiar la tabla antes de insertar nuevos valores
        for item in table.get_children():
            table.delete(item)

        # Insertar contraejemplos en la tabla
        if hasattr(cf, "cf_examples_list") and cf.cf_examples_list:
            cf_df = pd.DataFrame(cf.cf_examples_list[0].final_cfs_df, columns=features + ["aprobado"])
            for i, row in cf_df.iterrows():
                table.insert("", "end", values=(
                    studytime_map.get(row["studytime"], "Desconocido"),  # Convertir de categoría a horas reales
                    failures_map.get(row["failures"], "Desconocido"), 
                    row["absences"],
                    row["G1"],
                    row["G2"],
                    "Aprobado ✅" if row["aprobado"] == 1 else "Suspendido ❌"
                ))

        else:
            messagebox.showinfo("Info", "No se pudieron generar contraejemplos.")

    except ValueError:
        messagebox.showerror("Error", "Por favor, ingresa valores válidos.")

# Crear la interfaz
root = tk.Tk()
root.title("Predicción de Aprobar la asignatura de matemáticas en el tercer trimestre")

tk.Label(root, text="Horas de estudio:").grid(row=0, column=0)
entry_horas = tk.Entry(root)
entry_horas.grid(row=0, column=1)

tk.Label(root, text="Número de materias suspendidas:").grid(row=1, column=0)
entry_failures = tk.Entry(root)
entry_failures.grid(row=1, column=1)

tk.Label(root, text="Cantidad de faltas de asistencia (0-93):").grid(row=2, column=0)
entry_absences = tk.Entry(root)
entry_absences.grid(row=2, column=1)

tk.Label(root, text="Nota del primer trimestre (0-20):").grid(row=3, column=0)
entry_G1 = tk.Entry(root)
entry_G1.grid(row=3, column=1)

tk.Label(root, text="Nota del segundo trimestre (0-20):").grid(row=4, column=0)
entry_G2 = tk.Entry(root)
entry_G2.grid(row=4, column=1)

btn_predecir = tk.Button(root, text="Predecir", command=predecir)
btn_predecir.grid(row=5, column=0, pady=10)

btn_contrafactuales = tk.Button(root, text="Generar Contrafactuales", command=generar_contrafactuales)
btn_contrafactuales.grid(row=5, column=1, pady=10)

# Tabla para mostrar contraejemplos
columns = ("Horas Estudio", "Número de asignaturas suspensas", "Faltas de asistencia", "Nota del primer trimestre", "Nota del segundo trimestre", "Resultado")
table = ttk.Treeview(root, columns=columns, show="headings")
for col in columns:
    table.heading(col, text=col)
    table.column(col, width=100)

table.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()

