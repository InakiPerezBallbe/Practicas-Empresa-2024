import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder

def explainLime(df: pd.DataFrame, model, row: int, encoders: dict, target: str, num_features: int = None, class_index: int = None):
    # --- 1. Validación de Entrada y Preparación de Datos ---

    # Comprueba si el índice de la fila solicitada es válido para el DataFrame.
    if row < 0 or row >= len(df):
        raise IndexError(f"El índice de fila {row} está fuera del rango permitido (0 a {len(df) - 1}).")

    # Separa las características (X) de la variable objetivo (target).
    X = df.drop(columns=target)
    # Selecciona la fila específica que se va a explicar. Se usa .iloc[[row]] para asegurarse
    # de que el resultado sea un DataFrame de una fila, no una Serie.
    X_row = X.iloc[[row]].copy()

    # --- 2. Preparación de Metadatos para LIME ---
    # LIME necesita saber qué columnas son categóricas y cuáles son sus posibles valores.
    # Esta sección extrae esa información del diccionario 'encoders'.

    categorical_feature_names = [col for col in encoders.keys() if col in X.columns] # Nombres de las características categóricas originales
    categorical_feature_indices = [] # Índices de las características categóricas en el DataFrame X (codificado)
    categorical_names = {} # Diccionario que mapea un índice a los nombres de sus categorías

    # Itera sobre los codificadores para construir los metadatos para LIME.
    for col in categorical_feature_names:
        encoder = encoders[col]
        # Para OneHotEncoder, LIME no agrupa las columnas generadas. Por convención, se podría
        # tratar la primera columna OHE como la "representante" de la característica original.
        if isinstance(encoder, OneHotEncoder):
            ohe_cols = encoder.get_feature_names_out([col])
            # Comprueba si todas las columnas OHE esperadas existen en el DataFrame X.
            if all(c in X_row.columns for c in ohe_cols):
                # Se añade el índice de la primera columna OHE a la lista de características categóricas.
                # Esta es una simplificación; LIME tratará cada columna OHE como una característica binaria.
                categorical_feature_indices.append(X.columns.get_loc(ohe_cols[0]))
                # Se mapea el índice de esa primera columna OHE a la lista de categorías originales.
                categorical_names[X.columns.get_loc(ohe_cols[0])] = encoder.categories_[0].tolist()
        
        # Para OrdinalEncoder y LabelEncoder, la característica sigue siendo una sola columna.
        elif isinstance(encoder, (OrdinalEncoder, LabelEncoder)):
            # Se añade el índice de la columna a la lista de características categóricas.
            categorical_feature_indices.append(X.columns.get_loc(col))
            # Se extraen los nombres de las clases/categorías del codificador y se mapean al índice.
            if hasattr(encoder, "categories_"): # Para OrdinalEncoder
                categorical_names[X.columns.get_loc(col)] = encoder.categories_[0].tolist()
            elif hasattr(encoder, "classes_"): # Para LabelEncoder
                categorical_names[X.columns.get_loc(col)] = encoder.classes_.tolist()

    # --- 3. Configuración y Ejecución del Explicador LIME ---

    # Obtiene los nombres de las clases del modelo.
    class_names = model.classes_.tolist()
    # Determina para qué clase se generará la explicación.
    if class_index is None:
        # Si no se especifica, se usa la clase real de la fila que se está explicando.
        class_val = df[target].iloc[row]
        if class_val in class_names:
            class_index = class_names.index(class_val)
        else:
            raise ValueError(f"La clase '{class_val}' no está entre las clases del modelo: {class_names}")
    elif class_index not in range(len(class_names)):
        raise ValueError(f"El índice de clase {class_index} está fuera del rango válido (0 a {len(class_names)-1}).")

    # Inicializa el LimeTabularExplainer.
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values, # El conjunto de datos de entrenamiento (o una muestra representativa)
        feature_names=X.columns.tolist(), # Nombres de las columnas (pueden estar en formato OHE)
        class_names=class_names, # Nombres de las clases objetivo
        categorical_features=categorical_feature_indices, # Índices de las columnas categóricas
        categorical_names=categorical_names, # Mapeo de índices a nombres de categorías
        mode="classification",
        random_state=42
    )

    # Determina el número de características a mostrar en la explicación.
    if num_features is None:
        num_features = len(X.columns)

    # Genera la explicación para la instancia (fila) seleccionada.
    exp = explainer.explain_instance(
        data_row=X_row.values[0], # La fila a explicar, como un array 1D
        predict_fn=model.predict_proba, # La función de predicción del modelo
        num_features=num_features,
        labels=(class_index,) # Explicar solo para la clase de interés
    )

    # --- 4. Decodificación de la Fila Original para la Visualización ---
    # Esta sección decodifica la fila original para que las etiquetas del gráfico
    # puedan mostrar los valores originales legibles.
    fila_decodificada = X_row.copy()

    for col, encoder in encoders.items():
        if isinstance(encoder, LabelEncoder) and col in fila_decodificada.columns:
            fila_decodificada[col] = encoder.inverse_transform(fila_decodificada[col].astype(int))
        elif isinstance(encoder, OrdinalEncoder) and col in fila_decodificada.columns:
            fila_decodificada[col] = encoder.inverse_transform(fila_decodificada[[col]]).ravel()
        elif isinstance(encoder, OneHotEncoder):
            ohe_cols = encoder.get_feature_names_out([col])
            if all(c in fila_decodificada.columns for c in ohe_cols):
                decoded = encoder.inverse_transform(fila_decodificada[ohe_cols])[0][0]
                fila_decodificada.drop(columns=ohe_cols, inplace=True) # Eliminar columnas OHE
                fila_decodificada[col] = decoded # Añadir columna original decodificada

    # Convierte el DataFrame de una fila a una Serie para facilitar el acceso a los valores.
    fila_decodificada = fila_decodificada.squeeze()

    # --- 5. Preparación de Etiquetas para el Gráfico ---
    # Obtiene la lista de explicaciones (ej: 'edad <= 30') y sus importancias.
    exp_list = exp.as_list(label=class_index)
    features_names = [f[0] for f in exp_list]
    importance = [f[1] for f in exp_list]

    # Crea etiquetas más legibles para el gráfico.
    # Por ejemplo, si LIME devuelve 'sexo_male <= 0.50', esta lógica
    # intentará reemplazarlo con algo más claro como 'sexo = female'.
    etiquetas = []
    for f in features_names:
        match_col = None # Para guardar el nombre de la columna original si se encuentra una coincidencia
        # Intenta encontrar a qué columna original pertenece la característica de la explicación de LIME.
        for original_col in encoders.keys():
            # Comprueba si la explicación de LIME (f) empieza con el nombre de una columna original
            if f.startswith(original_col + '=') or f.startswith(original_col + '<') or f.startswith(original_col + '>'):
                match_col = original_col
                break
            # Comprueba si la explicación (f) empieza con el nombre de una columna generada por OHE.
            encoder = encoders[original_col]
            if isinstance(encoder, OneHotEncoder):
                for ohe_col in encoder.get_feature_names_out([original_col]):
                    if f.startswith(ohe_col):
                        match_col = original_col
                        break
        
        # Si se encontró una columna original, crea una etiqueta con el valor decodificado.
        if match_col and match_col in fila_decodificada:
            etiquetas.append(f"{match_col} = {fila_decodificada[match_col]}")
        else:
            # Si no, usa la etiqueta tal como la da LIME.
            etiquetas.append(f)

    # --- 6. Creación del Gráfico ---
    plt.figure(figsize=(12, 6))
    # Crea un gráfico de barras horizontales.
    plt.barh(etiquetas, importance, color=["red" if x < 0 else "green" for x in importance])
    plt.xlabel("Importancia (Peso de LIME)")
    plt.ylabel("Características")
    plt.title(f"Explicación LIME - fila {row} (para la clase '{class_names[class_index]}')")
    plt.yticks(fontsize=12) # Tamaño de la fuente para las etiquetas de las características
    plt.axvline(0, color="black", linewidth=0.8) # Línea vertical en cero para referencia
    plt.tight_layout() # Ajusta el gráfico para que todo quepa bien
    plt.show() # Muestra el gráfico


def explainShapLocal(df: pd.DataFrame, model, row: int, encoders: dict, target: str, num_features: int = None, class_index=None):
    """
    Genera y visualiza una explicación local de SHAP para una única instancia (fila) de un DataFrame.

    Args:
        df (pd.DataFrame): El DataFrame completo (características y objetivo).
        model (any): El modelo de clasificación entrenado (compatible con scikit-learn).
        row (int): El índice de la fila en el DataFrame a explicar.
        encoders (dict): Diccionario con los codificadores ajustados para decodificar características.
        target (str): El nombre de la columna objetivo.
        num_features (int, optional): Número máximo de características a mostrar en el gráfico.
        class_index (int, optional): El índice de la clase a explicar. Si es None, se infiere
                                     a partir del valor real de la fila.
    """
    
    # --- 1. Validación de Entrada y Preparación de Datos ---
    # Comprueba si el índice de la fila solicitada es válido para el DataFrame.
    if row < 0 or row >= len(df):
        raise IndexError(f"El índice de fila {row} está fuera del rango permitido (0 a {len(df) - 1}).")

    # Separa las características (X) eliminando la columna objetivo.
    X = df.drop(columns=target)
    # Selecciona la fila específica que se va a explicar como un DataFrame de una sola fila.
    X_row = X.iloc[[row]].copy()

    # --- 2. Inicialización del Explicador SHAP y Cálculo de Valores ---
    # Se crea un objeto Explainer de SHAP.
    # - `model.predict_proba`: Se le pasa la función de predicción de probabilidad del modelo.
    # - `X`: Se le pasa el conjunto de datos de características para que SHAP pueda aprender
    #        la distribución de fondo de los datos, lo cual es necesario para algunos explicadores.
    # - `feature_names`: Se le proporcionan los nombres de las columnas.
    explainer = shap.Explainer(model.predict_proba, X, feature_names=X.columns.tolist())
    
    # Se calculan los valores SHAP para la instancia específica (X_row).
    shap_values = explainer(X_row)

    # --- 3. Manejo de Clases para la Explicación ---
    # Determina para qué clase se deben mostrar los valores SHAP.
    if class_index is None:
        # Si no se especifica un índice de clase, se infiere a partir del valor real
        # de la variable objetivo para la fila que se está explicando.
        class_val = df[target].iloc[row]
        if class_val in list(model.classes_):
            class_index = list(model.classes_).index(class_val) # Encuentra el índice de la clase real
        else:
            raise ValueError(f"La clase real '{class_val}' no se encuentra entre las clases del modelo: {list(model.classes_)}")
    else:
        # Si se proporciona un índice de clase, se valida que esté dentro del rango correcto.
        if class_index not in range(len(model.classes_)):
            raise ValueError(f"El índice de clase {class_index} no está dentro del rango válido 0 a {len(model.classes_) - 1}.")

    # --- 4. Extracción de los Valores SHAP Relevantes ---
    # El objeto `shap_values` puede tener 2 o 3 dimensiones dependiendo del tipo de modelo y explicador.
    if len(shap_values.shape) == 3:
        # Para modelos multiclase, shap_values tiene forma (n_muestras, n_características, n_clases).
        # Se seleccionan los valores para la única muestra (índice 0) y para la clase de interés (class_index).
        values = shap_values.values[0, :, class_index]
    else:
        # Para modelos binarios, la forma suele ser (n_muestras, n_características).
        # Se seleccionan los valores para la única muestra (índice 0).
        values = shap_values.values[0, :]

    # Se obtienen los nombres de las características y se calcula la importancia absoluta.
    features = shap_values.feature_names
    importances = np.abs(values)

    # --- 5. Selección de Características Más Importantes para Visualizar ---
    if num_features is None:
        num_features = len(features)
    n_features_to_plot = min(num_features, len(features))
    
    # Se ordenan los índices de las características por su importancia (de mayor a menor)
    # y se seleccionan las 'n_features_to_plot' más importantes.
    sorted_idx = np.argsort(importances)[::-1][:n_features_to_plot]

    # --- 6. Decodificación de Nombres de Características para Legibilidad ---
    # Se crea un mapeo para traducir los nombres de las columnas (posiblemente codificadas)
    # a etiquetas legibles para el gráfico.
    feature_label_map = {}

    # Itera sobre los codificadores proporcionados para crear etiquetas decodificadas.
    for col, encoder in encoders.items():
        if isinstance(encoder, OneHotEncoder):
            # Para OHE, se encuentra el valor original y se crea una etiqueta para cada columna OHE generada.
            ohe_cols = encoder.get_feature_names_out([col])
            if all(c in X.columns for c in ohe_cols):
                decoded_val = encoder.inverse_transform(X_row[ohe_cols])[0][0]
                for c in ohe_cols:
                    feature_label_map[c] = f"{col} = {decoded_val}"
        elif isinstance(encoder, (LabelEncoder, OrdinalEncoder)):
            # Para LE/OE, se decodifica el valor numérico a su etiqueta de string original.
            if col in X_row:
                if isinstance(encoder, LabelEncoder):
                    val = encoder.inverse_transform([int(X_row[col].values[0])])[0]
                else: # OrdinalEncoder
                    val = encoder.inverse_transform(X_row[[col]])[0][0]
                feature_label_map[col] = f"{col} = {val}"

    # Se asegura de que las columnas numéricas no codificadas también tengan una etiqueta legible.
    for col in X_row.columns:
        if col not in feature_label_map:
            feature_label_map[col] = f"{col} = {X_row[col].values[0]:.2f}" # Formateado a 2 decimales

    # Se crean las etiquetas finales para el eje Y del gráfico, en el orden de importancia.
    y_labels = [feature_label_map.get(features[i], features[i]) for i in sorted_idx]

    # --- 7. Creación y Visualización del Gráfico SHAP Local ---
    plt.figure(figsize=(12, max(6, n_features_to_plot * 0.5))) # Tamaño de figura dinámico
    
    # Se crea un gráfico de barras horizontales.
    # El orden se invierte ([::-1]) para que la característica más importante aparezca arriba.
    plt.barh(
        y=np.array(y_labels)[::-1], # Etiquetas del eje Y
        width=values[sorted_idx][::-1], # Ancho de las barras (valor SHAP)
        color=["green" if v > 0 else "red" for v in values[sorted_idx][::-1]] # Verde si empuja hacia la clase, rojo si aleja
    )
    
    plt.xlabel("Contribución SHAP (Impacto en la predicción del modelo)")
    plt.title(f"Explicación SHAP Local para la Fila {row} (hacia la clase '{model.classes_[class_index]}')")
    plt.axvline(0, color="black", linewidth=0.8) # Línea en cero para referencia visual
    # La línea `invert_yaxis()` se elimina porque la inversión ya se hizo al pasar los datos al gráfico.
    plt.tight_layout() # Ajusta el gráfico para que todo quepa bien
    plt.show()

def explainShapGlobal(df: pd.DataFrame, model, encoders: dict, target: str, class_index=None):
    # Se eliminan las columnas objetivo para obtener solo las variables independientes
    X = df.drop(columns=target)

    # Se crea un explicador SHAP usando la función predict_proba del modelo (para clasificación probabilística)
    explainer = shap.Explainer(model.predict_proba, X)

    # Se obtienen los valores SHAP para todo el dataset
    shap_values = explainer(X)

    # Si el modelo es multiclase, se verifica qué índice de clase se debe analizar
    if class_index is None:
        class_index = 0  # Por defecto, se elige la clase 0
    elif class_index not in range(len(model.classes_)):
        raise ValueError(f"Índice de clase inválido: {class_index}. Debe estar entre 0 y {len(model.classes_) - 1}")

    # Si hay más de una clase (forma tridimensional), se extraen los valores SHAP para la clase deseada
    if len(shap_values.shape) == 3:
        values = shap_values.values[:, :, class_index]
    else:
        values = shap_values.values  # Para clasificación binaria o regresión

    # Se obtienen los nombres de las características
    features = shap_values.feature_names

    # Se crea un DataFrame con los valores SHAP absolutos, para evaluar la importancia
    shap_df = pd.DataFrame(np.abs(values), columns=features)

    # Se crea un diccionario para mapear las columnas codificadas a sus nombres originales
    col_mapping = {}

    for col, encoder in encoders.items():
        if isinstance(encoder, OneHotEncoder):
            # Para OneHotEncoder, se obtienen los nombres codificados (e.g., "sexo_m", "sexo_f") y se asignan al nombre original (e.g., "sexo")
            ohe_cols = encoder.get_feature_names_out([col])
            for c in ohe_cols:
                col_mapping[c] = col
        else:
            # Para LabelEncoder u OrdinalEncoder, la columna codificada conserva su nombre
            col_mapping[col] = col

    # Para cualquier otra columna no codificada, se conserva el nombre tal cual
    for col in shap_df.columns:
        if col not in col_mapping:
            col_mapping[col] = col

    # Se renombran las columnas del DataFrame SHAP con sus nombres originales
    shap_df.columns = [col_mapping[c] for c in shap_df.columns]

    # Se agrupan los valores SHAP por columna original y se suman (en caso de OHE con varias columnas)
    grouped = shap_df.groupby(shap_df.columns, axis=1).sum()

    # Se calcula la importancia media por columna (valor absoluto promedio)
    mean_importance = grouped.mean().sort_values(ascending=False)

    # Se genera un gráfico de barras horizontales con las importancias ordenadas
    plt.figure(figsize=(10, max(5, 0.4 * len(mean_importance))))
    plt.barh(mean_importance.index[::-1], mean_importance.values[::-1], color="skyblue")
    plt.xlabel("Importancia SHAP media (|valor|)")
    plt.title(f"Importancia Global SHAP - Clase {class_index}")
    plt.tight_layout()
    plt.show()