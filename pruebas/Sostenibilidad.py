import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.Preprocessing import readDataframe, standarize, oversample
from scripts.Encoding import encode
from scripts.Modeling import Modeling
from scripts.Explanation import explainLime, explainShapLocal, explainShapGlobal
from scripts.Counterfactual import Counterfactual
import pandas as pd

#PREPROCESAMIENTO
df = readDataframe("C:/Users/Usuario/OneDrive/Desktop/Practicas-Empresa-2024/Practicas-Empresa-2024/data/Sostenibilidad_tic.csv", 'utf-8', ",")

df.drop(columns="Marca_Temporal", axis=1, inplace=True)

df = standarize(df, "Grado", "C:/Users/Usuario/OneDrive/Desktop/Practicas-Empresa-2024/Practicas-Empresa-2024/standard/Grado.txt")


criterios = []
criterios.append(df['Frecuencia_Uso_Dispositivos'].str.lower().isin(['nunca', 'ocasionalmente']))
criterios.append(df['Redes_Sociales_Diario'].str.lower().isin(['nunca', 'ocasionalmente']))
criterios.append(df['Tiempo_Redes_Sociales'].str.lower().isin(['menos de 1 hora', 'entre 1 hora y 2 horas']))
criterios.append(df['Conocimiento_Huella_Carbono'].str.lower().isin(['si']))
criterios.append(df['Impacto_Uso_Dispositivos'].str.lower().isin(['sí, ocasionalmente lo considero.', 'sí, a menudo pienso en ello.']))
criterios.append(df['Apagar_Dispositivos'].str.lower().isin(['siempre apago mis dispositivos.', 'la mayoría de las veces.']))
criterios.append(df['Enviar_Correos_Pesados'].str.lower().isin(['siempre.', 'la mayoría de las veces.']))
criterios.append(df['Imprimir_Grandes_Documentos'].str.lower().isin(['siempre.', 'la mayoría de las veces.']))
criterios.append(df['Comprimir_Archivos'].str.lower().isin(['siempre.', 'la mayoría de las veces.']))
criterios.append(df['Reducir_Redes_Sociales'].str.lower().isin(['sí, definitivamente.', 'sí, pero me resulta difícil.']))
criterios.append(df['Impacto_CO2_Stickers_Gifs'] == 'no, ya conocía esta información.')
criterios.append(df['Reducir_Stickers_Gifs'] == 'sí, estoy dispuesto/a.')
criterios.append(df['Aprendizaje_Sostenibilidad_TIC'] == 'sí, definitivamente.')
criterios.append(df['Compartir_Conocimiento_Sostenibilidad'] == 'sí, definitivamente.')

df["Conciencia_Ambiental"] = (sum(criterios) >= 8).astype(int)
#df = oversample(df, "Conciencia_Ambiental")

df.to_csv("datos.csv")

df, encoders = encode(df, "C:/Users/Usuario/OneDrive/Desktop/Practicas-Empresa-2024/Practicas-Empresa-2024/encoding/Sostenibilidad.txt")

m = Modeling(df, "Conciencia_Ambiental")
m.train("AdaBoost")

explainShapGlobal(df, m.model, encoders, "Conciencia_Ambiental", class_index=1)

"""

df = pd.DataFrame(pp.data.iloc[[0], pp.data.columns != "Conciencia_Ambiental"])
df["Genero"] = 1
df["Edad"] = 2.0
df["Impacto_Uso_Dispositivos"] = 1.0



a, b = c.predict(df)

print(a)
print(b)

df = pd.DataFrame(pp.data.iloc[[0]])
df["Genero"] = 1
df["Edad"] = 2.0

cf = Counterfactual(c.model, pp.data, "Conciencia_Ambiental")

a, b = cf.counterfac(df, encoders, 1)"""
