from Analysis import Preprocessing,  Classification
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

link = "C:/Users/Usuario/OneDrive/Desktop/Practicas-Empresa-2024/Practicas-Empresa-2024/data/Eslogans.csv"

pp = Preprocessing(link, "latin1", ";")
pp.delete("Marca")

encoders = pp.encode("encoding/Eslogans.txt")

for col, encoder in encoders.items():
    if isinstance(encoder, CountVectorizer):
        transformed = encoder.transform(["Para ella, todo debe ser perfecto, porque una mujer siempre quiere más."])
        series_cv = pd.Series(
            transformed.toarray().flatten(), 
            index=encoder.get_feature_names_out()
        )
        break

c = Classification(pp.data, "Sesgo")
c.train(["XGBoost"])

print(c.predict(series_cv))


