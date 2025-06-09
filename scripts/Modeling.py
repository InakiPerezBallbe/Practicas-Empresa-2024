import sys
import os
from typing import Tuple
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
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

def chooseModel(df: pd.DataFrame, target: str, test_size=0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, any]:

    Y = df[target]
    X = df.drop(target, axis=1)
    X.columns = X.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=test_size, random_state=42)

    # Detectar clases presentes
    clases_existentes = np.unique(ytrain)
    min_clase = min(np.bincount(ytrain))
    n_splits = min(3, min_clase) if min_clase >= 2 else 2
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Definir métricas
    metricas = {
        'Precision': 'accuracy',
        'Exactitud': make_scorer(precision_score, average='macro', zero_division=0, labels=clases_existentes),
        'Recall': make_scorer(recall_score, average='macro', zero_division=0, labels=clases_existentes),
        'F1 Score': make_scorer(f1_score, average='macro', zero_division=0, labels=clases_existentes)
    }

    # Definir clasificadores
    modelos = {
        'Naive Bayes': GaussianNB(),
        'Árboles de Decisión': DecisionTreeClassifier(random_state=42),
        'Redes Neuronales': MLPClassifier(max_iter=1000, random_state=42),
        "Lineales": LogisticRegression(max_iter=1000),
        "Support Vector Machine": SVC(kernel='rbf', probability=True),
        "Neighbors": KNeighborsClassifier()
    }
    
    __metricsTable(modelos, metricas, xtrain, ytrain, cv)
    nombre_modelo, modelo_seleccionado = __selectModel(modelos)

    if nombre_modelo == "Naive Bayes":
        modelos = {
        'GaussianNB': GaussianNB(),
        'MultinomialNB': MultinomialNB(),
        'BernoulliNB': BernoulliNB(),
        'ComplementNB': ComplementNB(),
        'CategoricalNB': CategoricalNB()
        }

    elif nombre_modelo == "Árboles de Decisión":
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

    elif nombre_modelo == "Lineales":
        modelos = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SGDClassifier (hinge)": SGDClassifier(loss="hinge", max_iter=1000),
        "SGDClassifier (log)": SGDClassifier(loss="log_loss", max_iter=1000),
        "RidgeClassifier": RidgeClassifier(),
        "Perceptron": Perceptron(),
        "PassiveAggressiveClassifier": PassiveAggressiveClassifier(),
        }

    elif nombre_modelo == "Support Vector Machine":
        modelos = {
        "SVC (RBF kernel)": SVC(kernel='rbf', probability=True),
        "LinearSVC": LinearSVC(),
        "NuSVC": NuSVC(probability=True)
        }

    elif nombre_modelo == "Neighbors":
        modelos = {
        "KNeighborsClassifier": KNeighborsClassifier(),
        "RadiusNeighborsClassifier": RadiusNeighborsClassifier(radius=10.0),
        "NearestCentroid": NearestCentroid()
        }
    
    __metricsTable(modelos, metricas, xtrain, ytrain, cv)
    nombre_modelo, modelo_seleccionado = __selectModel(modelos)

    return xtrain, xtest, ytrain, ytest, modelo_seleccionado

def __metricsTable(modelos: dict[str, any], metricas: dict[str, any], X: pd.DataFrame, Y: pd.Series, cv: StratifiedKFold):
    print(f"\n🔍 Evaluando modelos")
    tabla = pd.DataFrame(columns=metricas)

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

    tabla.index.name = "Modelo"
    print("\n📊 Resultados detallados (media ± desviación típica):")
    print(tabulate(tabla, headers="keys", tablefmt="fancy_grid", stralign="center", numalign="center"))

def __selectModel(modelos: dict[str, any]) -> Tuple[str, any]:
    # Mostrar menú de selección
    nombres_modelos = list(modelos.keys())
    print("\n🔽 Selecciona un modelo para continuar:")
    for i, nombre in enumerate(nombres_modelos, 1):
        print(f"{i}. {nombre}")
    
    while True:
        try:
            seleccion = int(input("Introduce el número del modelo que deseas seleccionar: "))
            if 1 <= seleccion <= len(nombres_modelos):
                print(f"\n✅ Has seleccionado: {nombres_modelos[seleccion - 1]}")
                return nombres_modelos[seleccion - 1], modelos[nombres_modelos[seleccion - 1]]
            else:
                print("⚠️ Opción no válida. Elige un número del 1 al", len(nombres_modelos))
        except ValueError:
            print("⚠️ Entrada no válida. Introduce un número entero.")

def classify(model: any, row: pd.Series):
    ypred = model.predict(row)
    yprob = model.predict_proba(row)
    prob_values = [yprob[i][ypred[i]] for i in range(len(ypred))]
    
    row_pred = pd.concat([row, ypred], axis=1)
    
    prob_values_percent = [p * 100 for p in prob_values]
    prob_series = pd.Series(prob_values_percent, name="Predicted_Probability_%").round(2)

    return row_pred, prob_series
