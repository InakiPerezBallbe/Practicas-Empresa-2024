#pip install --upgrade numpy
#pip install --upgrade pandas
#pip install dice-ml

import dice_ml
from dice_ml.utils import helpers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Cargar los datos
dataset = helpers.load_adult_income_dataset()
print(f"Datos cargados: {dataset.shape}")
print(dataset)

# Comprobar los valores únicos en 'income' antes de la conversión
print(f"Valores únicos en 'income' antes de la conversión: {dataset['income'].unique()}")

# Eliminar filas con NaN en "income"
dataset.dropna(subset=["income"], inplace=True)
print(f"Datos después de eliminar NaN en 'income': {dataset.shape}")

# Convertir variables categóricas a números
categorical_columns = ["workclass", "education", "marital_status", "occupation", "race", "gender"]
label_encoders = {}

for col in categorical_columns:
    if col in dataset.columns:
        le = LabelEncoder()
        dataset[col] = le.fit_transform(dataset[col].astype(str))  # Convertimos a string antes de transformar
        label_encoders[col] = le
    else:
        print(f"Advertencia: La columna '{col}' no está en el dataset.")

print(dataset)

# Eliminar cualquier otro NaN que pueda haber quedado
dataset.dropna(inplace=True)
print(f"Datos después de eliminar otros NaN: {dataset.shape}")

#Verificar si el dataset sigue teniendo datos antes de continuar
if dataset.shape[0] == 0:
    raise ValueError("Error: No hay datos después de la limpieza. Revisa los pasos anteriores.")

# Dividir en entrenamiento y prueba
train, test = train_test_split(dataset, test_size=0.2, random_state=42)
print(f"Datos de entrenamiento: {train.shape}")
print(f"Datos de prueba: {test.shape}")

X_train = train.drop(columns=["income"])
y_train = train["income"]
X_test = test.drop(columns=["income"])
y_test = test["income"]

# Verificar si hay NaN en y_train antes de entrenar
if y_train.isnull().sum() > 0:
    print("Warning: Se encontraron NaN en y_train. Eliminando...")
    y_train.dropna(inplace=True)

# Entrenar el modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Crear objeto de modelo DiCE
dice_model = dice_ml.Model(model=model, backend="sklearn")

# Especificar los tipos de variables
d = dice_ml.Data(dataframe=dataset, continuous_features=['age', 'hours_per_week'], outcome_name='income')

# Crear el generador de contraejemplos
exp = dice_ml.Dice(d, dice_model, method="random")

# Seleccionar una instancia para generar contraejemplos
query_instance = X_test.iloc[0:1]  # Tomamos la primera fila del test

# Generar contraejemplos
cf = exp.generate_counterfactuals(query_instance, total_CFs=3, desired_class="opposite")

# Visualizar los resultados
cf.visualize_as_dataframe()