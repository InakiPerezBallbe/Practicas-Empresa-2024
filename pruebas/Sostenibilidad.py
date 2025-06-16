import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.Preprocessing import readDataframe, standarize, oversample
from scripts.Encoding import encode
from scripts.Modeling import chooseModel
from scripts.Explanation import explainLime, explainShapLocal, explainShapGlobal
from scripts.Counterfactual import generate_counterfactuals
import pandas as pd

df = readDataframe("./data/Sostenibilidad_tic.csv", "utf-8", ",")
df = standarize(df, "Grado", "./standard/Grado.txt")
df.loc[(df["Especialidad"] == "matemáticas") &
         (df["Grado"].str.lower() == "ingeniería"), "Grado"] = "grado en ingeniería matemática"

# Crear una lista de condiciones comprometidas
criterios = []
criterios.append(df['Frecuencia_Uso_Dispositivos'].isin(['nunca', 'ocasionalmente']))
criterios.append(df['Redes_Sociales_Diario'].isin(['nunca', 'ocasionalmente']))
criterios.append(df['Tiempo_Redes_Sociales'].isin(['menos de 1 hora', 'entre 1 hora y 2 horas']))
criterios.append(df['Conocimiento_Huella_Carbono'].isin(['si']))
criterios.append(df['Impacto_Uso_Dispositivos'].isin(['sí, ocasionalmente lo considero.', 'sí, a menudo pienso en ello.']))
criterios.append(df['Apagar_Dispositivos'].isin(['siempre apago mis dispositivos.', 'la mayoría de las veces.']))
criterios.append(df['Enviar_Correos_Pesados'].isin(['siempre.', 'la mayoría de las veces.']))
criterios.append(df['Imprimir_Grandes_Documentos'].isin(['siempre.', 'la mayoría de las veces.']))
criterios.append(df['Comprimir_Archivos'].isin(['siempre.', 'la mayoría de las veces.']))
criterios.append(df['Reducir_Redes_Sociales'].isin(['sí, definitivamente.', 'sí, pero me resulta difícil.']))
criterios.append(df['Impacto_CO2_Stickers_Gifs'] == 'no, ya conocía esta información.')
criterios.append(df['Reducir_Stickers_Gifs'] == 'sí, estoy dispuesto/a.')
criterios.append(df['Aprendizaje_Sostenibilidad_TIC'] == 'sí, definitivamente.')
criterios.append(df['Compartir_Conocimiento_Sostenibilidad'] == 'sí, definitivamente.')

# Crear columna 'conciencia_ambiental'
df['Conciencia_Ambiental'] = (sum(criterios) >= 8).astype(int)

df.drop(columns="Marca_Temporal", inplace=True)

df = oversample(df, "Conciencia_Ambiental")

df, encoders = encode(df, "./encoding/Sostenibilidad.txt")

target = "Conciencia_Ambiental"
xtrain, xtest, ytrain, ytest, model = chooseModel(df, target)

explainShapGlobal(df, model, encoders, target, 1)

"""
generate_counterfactuals(model, df, target, pd.DataFrame(df.iloc[[0]]), encoders, target_value=1)
"""
