import sys
import os
from typing import Tuple
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from tabulate import tabulate

import warnings
warnings.filterwarnings("ignore", message=".*within_class_std_dev_ has at least 1 zero standard deviation.*")

def chooseModel(df: pd.DataFrame, target: str, test_size=0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, any]:
    # --- 1. Preparación Inicial de Datos ---
    # Separa las características (X) de la variable objetivo (Y).
    Y = df[target]
    X = df.drop(target, axis=1)
    
    # Divide los datos en conjuntos de entrenamiento y prueba.
    # random_state=42 asegura que la división sea la misma cada vez que se ejecuta.
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=test_size, random_state=42)

    # --- 2. Configuración de la Validación Cruzada ---
    # Detecta las clases únicas presentes en el conjunto de entrenamiento.
    clases_existentes = np.unique(ytrain)
    # Cuenta el número de muestras en la clase más pequeña (minoritaria).
    min_clase = min(np.bincount(ytrain))
    
    # Determina el número de 'splits' (pliegues) para la validación cruzada.
    # Será 3 o el tamaño de la clase minoritaria (lo que sea menor), pero al menos 2.
    # Esto evita errores si la clase minoritaria es muy pequeña.
    n_splits = min(3, min_clase) if min_clase >= 2 else 2
    
    # StratifiedKFold es una estrategia de validación cruzada que mantiene la proporción
    # de las clases en cada pliegue, lo cual es muy importante para datos desbalanceados.
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # --- 3. Definición de Métricas de Evaluación ---
    # Se crea un diccionario de métricas que se usarán para evaluar los modelos.
    # Nota: El nombre 'Precision' se ha asignado a 'accuracy', y 'Exactitud' a 'precision_score'.
    # Esto puede ser confuso y debería revisarse.
    metricas = {
        'Precision': 'accuracy', # Esto en realidad es Accuracy (Exactitud)
        'Exactitud': make_scorer(precision_score, average='macro', zero_division=0, labels=clases_existentes), # Esto es Precision
        'Recall': make_scorer(recall_score, average='macro', zero_division=0, labels=clases_existentes),
        'F1 Score': make_scorer(f1_score, average='macro', zero_division=0, labels=clases_existentes)
    }

    # --- 4. Definición de Categorías de Modelos (Primera Fase) ---
    # Un diccionario con categorías amplias de modelos a evaluar inicialmente.
    modelos = {
        'Naive Bayes': GaussianNB(),
        'Árboles de Decisión': DecisionTreeClassifier(random_state=42),
        'Redes Neuronales': MLPClassifier(max_iter=1000, random_state=42),
        "Lineales": LogisticRegression(max_iter=1000),
        "Support Vector Machine": SVC(kernel='rbf', probability=True),
        "Neighbors": KNeighborsClassifier()
    }
    
    # --- Función Interna para Evaluar y Mostrar Resultados ---
    def metricsTable(modelos: dict[str, any], metricas: dict[str, any], X: pd.DataFrame, Y: pd.Series, cv: StratifiedKFold):
        print(f"\n🔍 Evaluando modelos...")
        tabla = pd.DataFrame(columns=metricas)

        for nombre, modelo in modelos.items():
            try:
                # 👉 Verificación y conversión especial si se selecciona CatBoostClassifier
                if isinstance(modelo, CatBoostClassifier):
                    X_copy = X.astype(str).copy()
                    Y_copy = Y.astype(str).copy()
                    modelo.set_params(cat_features=list(X_copy.columns))
                    resultados = cross_validate(modelo, X_copy, Y_copy, cv=cv, scoring=metricas, error_score='raise')
                else:
                    # `cross_validate` entrena y evalúa el modelo usando la estrategia de validación cruzada 'cv'.
                    resultados = cross_validate(modelo, X, Y, cv=cv, scoring=metricas, error_score='raise')
                # Crea una fila con la media y la desviación estándar de los resultados de cada métrica.
                fila = {
                    metrica: f"{np.mean(resultados[f'test_{metrica}']):.4f} ± {np.std(resultados[f'test_{metrica}']):.4f}"
                    for metrica in metricas
                }
                tabla.loc[nombre] = fila
            except Exception as e:
                print(f"⚠️ Error al evaluar {nombre}: {e}")

        tabla.index.name = "Modelo"
        print("\n📊 Resultados detallados (media ± desviación típica):")
        # Imprime la tabla de resultados en un formato legible en la consola.
        print(tabulate(tabla, headers="keys", tablefmt="fancy_grid", stralign="center", numalign="center"))

    # --- Función Interna para que el Usuario Seleccione un Modelo ---
    def selectModel(modelos: dict[str, any]) -> Tuple[str, any]:
        nombres_modelos = list(modelos.keys())
        print("\n🔽 Selecciona un modelo para continuar:")
        for i, nombre in enumerate(nombres_modelos, 1):
            print(f"{i}. {nombre}")
        
        while True:
            try:
                seleccion = int(input("Introduce el número del modelo que deseas seleccionar: "))
                if 1 <= seleccion <= len(nombres_modelos):
                    nombre_seleccionado = nombres_modelos[seleccion - 1]
                    print(f"\n✅ Has seleccionado: {nombre_seleccionado}")
                    return nombre_seleccionado, modelos[nombre_seleccionado]
                else:
                    print("⚠️ Opción no válida. Elige un número del 1 al", len(nombres_modelos))
            except ValueError:
                print("⚠️ Entrada no válida. Introduce un número entero.")

    # --- 5. Ejecución del Flujo de Selección ---

    # --- FASE 1: Selección de Categoría de Modelo ---
    # Evalúa las categorías generales de modelos.
    metricsTable(modelos, metricas, xtrain, ytrain, cv)
    # Pide al usuario que elija una categoría.
    nombre_modelo, modelo_seleccionado = selectModel(modelos)

    # --- FASE 2: Selección de Modelo Específico ---
    # Basándose en la categoría elegida, se define un nuevo diccionario 'modelos'
    # con modelos más específicos de esa categoría.
    if nombre_modelo == "Naive Bayes":
        modelos = { 'GaussianNB': GaussianNB(), 'MultinomialNB': MultinomialNB(), 'BernoulliNB': BernoulliNB(), 'ComplementNB': ComplementNB(), 'CategoricalNB': CategoricalNB() }
    elif nombre_modelo == "Árboles de Decisión":
        modelos = { 'Decision Tree': DecisionTreeClassifier(random_state=42), 'Random Forest': RandomForestClassifier(random_state=42), 'Extra Trees': ExtraTreesClassifier(random_state=42), 'Gradient Boosting': GradientBoostingClassifier(random_state=42), 'Hist Gradient Boosting': HistGradientBoostingClassifier(random_state=42), 'AdaBoost': AdaBoostClassifier(random_state=42), 'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42, enable_categorical=True), 'LightGBM': LGBMClassifier(random_state=42, verbose=-1), 'CatBoost': CatBoostClassifier(verbose=0, random_state=42, cat_features=[col for col in df.columns if col != 'Conciencia_Ambiental']) }
    elif nombre_modelo == "Lineales":
        modelos = { "LogisticRegression": LogisticRegression(max_iter=1000), "SGDClassifier (hinge)": SGDClassifier(loss="hinge", max_iter=1000), "SGDClassifier (log)": SGDClassifier(loss="log_loss", max_iter=1000) }
    elif nombre_modelo == "Support Vector Machine":
        modelos = { "SVC (RBF kernel)": SVC(kernel='rbf', probability=True), "NuSVC": NuSVC(probability=True) }
    elif nombre_modelo == "Neighbors":
        modelos = { "KNeighborsClassifier": KNeighborsClassifier(), "RadiusNeighborsClassifier": RadiusNeighborsClassifier(radius=50.0) }
    
    if nombre_modelo != "Redes Neuronales":
        if isinstance(modelo_seleccionado, CatBoostClassifier):
            xtrain= xtrain.astype(int).copy()
            ytrain = ytrain.astype(int).copy()
        # Evalúa la nueva lista de modelos específicos.
        metricsTable(modelos, metricas, xtrain, ytrain, cv)
        # Pide al usuario que elija el modelo final.
        nombre_modelo, modelo_seleccionado = selectModel(modelos)
    
    # --- 6. Entrenamiento del Modelo Final ---
    # Entrena el modelo final seleccionado por el usuario con TODO el conjunto de entrenamiento.
    # 👉 Verificación y conversión especial si se selecciona CatBoostClassifier
    
    modelo_seleccionado.fit(xtrain, ytrain)

    # --- 7. Devolución de Resultados ---
    # Devuelve los conjuntos de datos de entrenamiento/prueba y el modelo final entrenado.
    return xtrain, xtest, ytrain, ytest, modelo_seleccionado

# 'model: any' indica que se espera un objeto de modelo, pero sin especificar tipo.
# 'row: pd.Series' indica que se espera una Serie de pandas como entrada.
def classify(model: any, row: pd.Series):
    # --- PROBLEMA 1: Forma de la entrada ---
    # Los métodos .predict() y .predict_proba() de scikit-learn esperan una entrada 2D
    # (como un DataFrame o un array NumPy de forma [n_muestras, n_características]).
    # Pasar una Serie 1D ('row') directamente aquí causará un error.
    # Se debe convertir la Serie a un DataFrame de una sola fila antes.
    ypred = model.predict(row)
    yprob = model.predict_proba(row)
    
    # Esta línea extrae la probabilidad correspondiente a la clase que fue predicha.
    # Por ejemplo, si yprob es [[0.1, 0.9]] y ypred es [1], seleccionará 0.9.
    prob_values = [yprob[i][ypred[i]] for i in range(len(ypred))]
    
    # --- PROBLEMA 2: Concatenación ---
    # pd.concat([row, ypred], axis=1) intentará concatenar una Serie (row)
    # con un array NumPy (ypred) por columnas, lo cual fallará o dará un resultado
    # inesperado. Se necesita que ambos sean DataFrames (o que se manejen los índices).
    row_pred = pd.concat([row, ypred], axis=1)
    
    # Convierte las probabilidades a formato de porcentaje.
    prob_values_percent = [p * 100 for p in prob_values]
    
    # Crea una Serie de pandas con las probabilidades en porcentaje y las redondea.
    prob_series = pd.Series(prob_values_percent, name="Predicted_Probability_%").round(2)

    return row_pred, prob_series
