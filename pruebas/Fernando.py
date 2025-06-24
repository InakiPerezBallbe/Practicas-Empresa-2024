import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.Preprocessing import readDataframe, oversample
from scripts.Modeling import chooseModel
from scripts.Explanation import explainShapGlobal

# Cargar el CSV
df = readDataframe("./data/datos_entrenamiento.csv", delimiter=",")

# Definir columnas a normalizar
columnas_a_normalizar = ['longitude', 'latitude', 'mean_power', 'distancia_m', 'profundidad_m']

# Inicializar y aplicar el escalador
scaler = StandardScaler()
df[columnas_a_normalizar] = scaler.fit_transform(df[columnas_a_normalizar])

df.drop(columns=["site_id", "longitude", "latitude"], inplace=True)

#print(df["optimo"].value_counts())

target = "optimo"
xtrain, xtest, ytrain, ytest, model = chooseModel(df, target)

explainShapGlobal(df, model, None, target, 1)
