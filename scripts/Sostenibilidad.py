from Analysis import Preprocessing
from Analysis import Clasification
from Analysis import Counterfactual

import pandas as pd

#PREPROCESAMIENTO
pp = Preprocessing("C:/Users/Usuario/OneDrive/Desktop/Practicas-Empresa-2024/Practicas-Empresa-2024/data/Sostenibilidad_tic.csv")

pp.delete(["Marca_Temporal"])
pp.standarize("Grado", "C:/Users/Usuario/OneDrive/Desktop/Practicas-Empresa-2024/Practicas-Empresa-2024/standard/Grado.txt")
pp.replace("Especialidad", "Matemáticas", "Grado", "ingeniería", "Grado en Ingeniería Matemática")
pp.replace("Especialidad", "Otra", "Grado", "ingeniería", "Grado en Ingeniería (sin especificar)")

criterios = []

criterios.append(pp.data['Frecuencia_Uso_Dispositivos'].str.lower().isin(['nunca', 'ocasionalmente']))
criterios.append(pp.data['Redes_Sociales_Diario'].str.lower().isin(['nunca', 'ocasionalmente']))
criterios.append(pp.data['Tiempo_Redes_Sociales'].str.lower().isin(['menos de 1 hora', 'entre 1 hora y 2 horas']))
criterios.append(pp.data['Conocimiento_Huella_Carbono'].str.lower().isin(['si']))
criterios.append(pp.data['Impacto_Uso_Dispositivos'].str.lower().isin(['sí, ocasionalmente lo considero.', 'sí, a menudo pienso en ello.']))
criterios.append(pp.data['Apagar_Dispositivos'].str.lower().isin(['siempre apago mis dispositivos.', 'la mayoría de las veces.']))
criterios.append(pp.data['Enviar_Correos_Pesados'].str.lower().isin(['siempre.', 'la mayoría de las veces.']))
criterios.append(pp.data['Imprimir_Grandes_Documentos'].str.lower().isin(['siempre.', 'la mayoría de las veces.']))
criterios.append(pp.data['Comprimir_Archivos'].str.lower().isin(['siempre.', 'la mayoría de las veces.']))
criterios.append(pp.data['Reducir_Redes_Sociales'].str.lower().isin(['sí, definitivamente.', 'sí, pero me resulta difícil.']))
criterios.append(pp.data['Impacto_CO2_Stickers_Gifs'] == 'no, ya conocía esta información.')
criterios.append(pp.data['Reducir_Stickers_Gifs'] == 'sí, estoy dispuesto/a.')
criterios.append(pp.data['Aprendizaje_Sostenibilidad_TIC'] == 'sí, definitivamente.')
criterios.append(pp.data['Compartir_Conocimiento_Sostenibilidad'] == 'sí, definitivamente.')

pp.add("Conciencia_Ambiental", (sum(criterios) >= 8).astype(int))
encoders = pp.encode("C:/Users/Usuario/OneDrive/Desktop/Practicas-Empresa-2024/Practicas-Empresa-2024/categorical_cols/Sostenibilidad.txt")

c = Clasification(pp.data, "Conciencia_Ambiental", 0.2)
c.train(["K-Nearest Neighbors"])

df = pd.DataFrame(pp.data.iloc[[0], pp.data.columns != "Conciencia_Ambiental"])
df["Genero"] = 1
df["Edad"] = 2.0

df, prob = c.predict(df)

cf = Counterfactual(c.model, pp.data, "Conciencia_Ambiental")
cf.counterfac(df, encoders)
