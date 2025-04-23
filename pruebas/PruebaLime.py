#pip install lime
#pip install scikit-learn
#pip install numpy
#pip install scipy
#pip install matplotlib
#pip install ipython

import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1️⃣ Cargar datos
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
data = pd.read_csv("C:/Users/Usuario/OneDrive/Desktop/Practicas-Empresa-2024/Practicas-Empresa-2024/student-mat.csv", sep=";")

# Selección de características relevantes
features = ["studytime", "failures", "absences", "G1", "G2"]
target = "G3"

# Crear una variable de aprobado (1 si G3 >= 10, sino 0)
data["aprobado"] = (data["G3"] >= 10).astype(int)
X = data[features]
y = data["aprobado"]

# 2️⃣ Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **Normalizar los datos (opcional, pero recomendado para LIME)**
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

# 3️⃣ Entrenar modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4️⃣ Inicializar explicador LIME
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns,
    class_names=["Reprobado", "Aprobado"],
    mode="classification"
)

# 5️⃣ Ingresar datos de un nuevo estudiante
print("Ingrese los datos del nuevo estudiante:")
horas_estudio = float(input("Horas de estudio por semana (1-4): "))
failures = int(input("Cantidad de materias reprobadas anteriormente: "))
absences = int(input("Cantidad de faltas: "))
G1 = float(input("Calificación del primer período (0-20): "))
G2 = float(input("Calificación del segundo período: "))

# 6️⃣ Crear DataFrame con los valores ingresados
new_student = pd.DataFrame([[horas_estudio, failures, absences, G1, G2]], columns=features)

# 7️⃣ Obtener predicción del modelo para el nuevo estudiante
pred_proba = model.predict_proba(new_student)
pred_clase = model.predict(new_student)[0]  # 0 = Reprobado, 1 = Aprobado

# 8️⃣ Explicar la predicción con LIME
exp = explainer.explain_instance(new_student.iloc[0], model.predict_proba, num_features=5)

# 9️⃣ Mostrar el resultado de la predicción
resultado = "aprobará" if pred_clase == 1 else "suspenderá"
print(f"\nEl estudiante ha **{resultado}** con una probabilidad del {round(pred_proba[0][pred_clase] * 100, 2)}%.")

# 🔟 Mostrar resultados de forma visual
exp_list = exp.as_list()
features_names = [f[0] for f in exp_list]
importance = [f[1] for f in exp_list]

plt.figure(figsize=(12, 6))
plt.barh(features_names, importance, color=["red" if x < 0 else "green" for x in importance])
plt.xlabel("Importancia")
plt.ylabel("Características")
plt.title("Explicación de LIME para un estudiante")
plt.yticks(fontsize=12)
plt.axvline(0, color="black", linewidth=1)
plt.tight_layout()
plt.show()