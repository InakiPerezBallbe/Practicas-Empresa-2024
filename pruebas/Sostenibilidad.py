import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.Preprocessing import readDataframe, standarize, oversample
from scripts.Encoding import encode
from scripts.Modeling import chooseModel
from scripts.Explanation import explainLime, explainShapLocal, explainShapGlobal
from scripts.Counterfactual import Counterfactual
import pandas as pd


df = readDataframe("./data/Enc_Sostenibilidad.csv", "utf-8", ",")
target = "Conciencia_Ambiental"

model = chooseModel(df, target)


"""

cf = Counterfactual(model, df, target)

a, b = cf.counterfac(df, encoders, 1)
explainShapGlobal(df, m.model, encoders, "Conciencia_Ambiental", class_index=1)

df = pd.DataFrame(pp.data.iloc[[0], pp.data.columns != "Conciencia_Ambiental"])
df["Genero"] = 1
df["Edad"] = 2.0
df["Impacto_Uso_Dispositivos"] = 1.0

df = pd.DataFrame(pp.data.iloc[[0]])
df["Genero"] = 1
df["Edad"] = 2.0

"""
