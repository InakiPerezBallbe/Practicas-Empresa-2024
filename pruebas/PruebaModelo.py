import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from tabulate import tabulate

import warnings
warnings.filterwarnings("ignore", message=".*within_class_std_dev_ has at least 1 zero standard deviation.*")

# Cargar los datos
df = pd.read_csv("./data/Enc_Sostenibilidad.csv")
Y = df["Conciencia_Ambiental"]
X = df.drop("Conciencia_Ambiental", axis=1)
X.columns = X.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)

# Detectar clases presentes
clases_existentes = np.unique(Y)
min_clase = min(np.bincount(Y))
n_splits = min(3, min_clase) if min_clase >= 2 else 2
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Definir clasificadores
modelos = {
    'Naive Bayes': GaussianNB(),
    'Árboles de Decisión': DecisionTreeClassifier(random_state=42),
    'Redes Neuronales': MLPClassifier(max_iter=1000, random_state=42),
    "Lineales": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(kernel='rbf', probability=True),
    "Neighbors": KNeighborsClassifier()
}

# Definir métricas
metricas = {
    'Precision': 'accuracy',
    'Exactitud': make_scorer(precision_score, average='macro', zero_division=0, labels=clases_existentes),
    'Recall': make_scorer(recall_score, average='macro', zero_division=0, labels=clases_existentes),
    'F1 Score': make_scorer(f1_score, average='macro', zero_division=0, labels=clases_existentes)
}

# Guardar resultados
resultados_modelos = {}
print(f"\n🔍 Evaluando modelos")

for nombre, modelo in modelos.items():
    resultados = cross_validate(modelo, X, Y, cv=cv, scoring=metricas, error_score='raise')

    resumen = {}
    for metrica in metricas.keys():
        puntuaciones = resultados[f'test_{metrica}']
        media = np.mean(puntuaciones)
        std = np.std(puntuaciones)
        resumen[metrica] = f"{media:.4f} ± {std:.4f}"

    resultados_modelos[nombre] = resumen

# Crear DataFrame con los resultados
resultados_df = pd.DataFrame(resultados_modelos).T
resultados_df.index.name = "Modelo"

# Mostrar la tabla con columnas centradas
print("\n📊 Resultados detallados (media ± desviación típica):")
print(tabulate(resultados_df, headers="keys", tablefmt="fancy_grid", stralign="center", numalign="center"))

# Mostrar menú de selección
nombres_modelos = list(modelos.keys())
print("\n🔽 Selecciona un modelo para continuar:")
for i, nombre in enumerate(nombres_modelos, 1):
    print(f"{i}. {nombre}")

# Leer selección del usuario
opcion_valida = False
while not opcion_valida:
    try:
        seleccion = int(input("Introduce el número del modelo que deseas seleccionar: "))
        if 1 <= seleccion <= len(nombres_modelos):
            modelo_seleccionado = nombres_modelos[seleccion - 1]
            print(f"\n✅ Has seleccionado: {modelo_seleccionado}")
            opcion_valida = True
        else:
            print("⚠️ Opción no válida. Elige un número del 1 al", len(nombres_modelos))
    except ValueError:
        print("⚠️ Entrada no válida. Introduce un número entero.")

if modelo_seleccionado == "Árboles de Decisión":
    print("\n🌳 Re-evaluando con modelos clásicos de árboles de decisión...")
    modelos = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Extra Trees': ExtraTreesClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
        'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
    }

    columnas_metricas = list(metricas.keys())
    tabla = pd.DataFrame(columns=columnas_metricas)

    for nombre, modelo in modelos.items():
        resultados = cross_validate(modelo, X, Y, cv=cv, scoring=metricas, error_score='raise')
        fila = {}
        for metrica in metricas.keys():
            media = np.mean(resultados[f'test_{metrica}'])
            std = np.std(resultados[f'test_{metrica}'])
            fila[metrica] = f"{media:.4f} ± {std:.4f}"
        tabla.loc[nombre] = fila

    tabla.columns.name = "Métricas"
    print("\n🌲 Resultados con modelos de árboles:")
elif modelo_seleccionado == "Lineales":
    print("\nEvaluando clasificadores lineales...\n")

    modelos = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SGDClassifier (hinge)": SGDClassifier(loss="hinge", max_iter=1000),
        "SGDClassifier (log)": SGDClassifier(loss="log_loss", max_iter=1000),
        "RidgeClassifier": RidgeClassifier(),
        "Perceptron": Perceptron(),
        "PassiveAggressiveClassifier": PassiveAggressiveClassifier(),
    }

    tabla = pd.DataFrame(columns=metricas)

    for nombre, modelo in modelos.items():
        resultados = cross_validate(modelo, X, Y, cv=cv, scoring=metricas, error_score='raise')
        fila = {
            metrica: f"{np.mean(resultados[f'test_{metrica}']):.3f} ± {np.std(resultados[f'test_{metrica}']):.3f}"
            for metrica in metricas
        }
        tabla.loc[nombre] = fila

    print("\nResultados de clasificadores lineales:")
elif modelo_seleccionado == "Naive Bayes":
    modelos = {
        'GaussianNB': GaussianNB(),
        'MultinomialNB': MultinomialNB(),
        'BernoulliNB': BernoulliNB(),
        'ComplementNB': ComplementNB(),
        'CategoricalNB': CategoricalNB()
    }

    tabla = pd.DataFrame(columns=metricas.keys())

    for nombre, modelo in modelos.items():
        try:
            resultados = cross_validate(modelo, X, Y, cv=cv, scoring=metricas, error_score='raise')
            fila = {
                metrica: f"{np.mean(resultados[f'test_{metrica}']):.4f} ± {np.std(resultados[f'test_{metrica}']):.4f}"
                for metrica in metricas
            }
            tabla.loc[nombre] = fila
        except Exception as e:
            print(f"⚠️ Error al evaluar {nombre}: {e}")

    print("\n📊 Resultados de modelos Naive Bayes:\n")
elif modelo_seleccionado == "Support Vector Machine":

    modelos = {
        "SVC (RBF kernel)": SVC(kernel='rbf', probability=True),
        "LinearSVC": LinearSVC(),
        "NuSVC": NuSVC(probability=True)
    }

    tabla = pd.DataFrame(columns=metricas.keys())

    for nombre, modelo in modelos.items():
        resultados = cross_validate(modelo, X, Y, cv=cv, scoring=metricas, error_score='raise')
        fila = {
            metrica: f"{np.mean(resultados[f'test_{metrica}']):.3f} ± {np.std(resultados[f'test_{metrica}']):.3f}"
            for metrica in metricas
        }
        tabla.loc[nombre] = fila

    # Centrar nombres de columnas
    tabla.columns.name = "Métricas"
elif modelo_seleccionado == "Neighbors":

    modelos = {
        "KNeighborsClassifier": KNeighborsClassifier(),
        "RadiusNeighborsClassifier": RadiusNeighborsClassifier(radius=10.0),
        "NearestCentroid": NearestCentroid()
    }

    tabla = pd.DataFrame(columns=metricas.keys())

    for nombre, modelo in modelos.items():
        resultados = cross_validate(modelo, X, Y, cv=cv, scoring=metricas, error_score='raise')
        fila = []
        for metrica in metricas:
            media = resultados[f'test_{metrica}'].mean()
            std = resultados[f'test_{metrica}'].std()
            fila.append(f"{media:.3f} ± {std:.3f}")
        tabla.loc[nombre] = fila

    tabla.columns.name = "Métrica"

tabla.index.name = "Modelo"
print(tabulate(tabla, headers="keys", tablefmt="fancy_grid", stralign="center", numalign="center"))