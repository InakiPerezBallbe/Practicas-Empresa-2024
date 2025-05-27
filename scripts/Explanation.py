import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder

def explainLime(df: pd.DataFrame, model, row: int, encoders: dict, target: str, num_features: int = None, class_index: int = None):

    if row < 0 or row >= len(df):
        raise IndexError(f"El índice de fila {row} está fuera del rango permitido (0 a {len(df) - 1}).")

    X = df.drop(columns=target)
    X_row = X.iloc[[row]].copy()

    # --- Identificar columnas categóricas ---
    categorical_feature_names = [col for col in encoders.keys() if col in X.columns]
    categorical_feature_indices = []
    categorical_names = {}

    for col in categorical_feature_names:
        encoder = encoders[col]
        if isinstance(encoder, OneHotEncoder):
            ohe_cols = encoder.get_feature_names_out([col])
            if all(c in X_row.columns for c in ohe_cols):
                categorical_feature_indices.append(X.columns.get_loc(ohe_cols[0]))  # una por convención
                categorical_names[X.columns.get_loc(ohe_cols[0])] = encoder.categories_[0].tolist()
        elif isinstance(encoder, (OrdinalEncoder, LabelEncoder)):
            categorical_feature_indices.append(X.columns.get_loc(col))
            if hasattr(encoder, "categories_"):
                categorical_names[X.columns.get_loc(col)] = encoder.categories_[0].tolist()
            elif hasattr(encoder, "classes_"):
                categorical_names[X.columns.get_loc(col)] = encoder.classes_.tolist()

    # --- LIME Explainer ---
    class_names = model.classes_.tolist()
    if class_index is None:
        class_val = df[target].iloc[row]
        if class_val in class_names:
            class_index = class_names.index(class_val)
        else:
            raise ValueError(f"La clase '{class_val}' no está entre las clases del modelo: {class_names}")
    elif class_index not in range(len(class_names)):
        raise ValueError(f"El índice de clase {class_index} está fuera del rango válido (0 a {len(class_names)-1}).")

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns.tolist(),
        class_names=class_names,
        categorical_features=categorical_feature_indices,
        categorical_names=categorical_names,
        mode="classification",
        random_state=42
    )

    if num_features is None:
        num_features = len(X.columns)

    exp = explainer.explain_instance(
        data_row=X_row.values[0],
        predict_fn=model.predict_proba,
        num_features=num_features,
        labels=(class_index,)
    )

    # --- Decodificar valores originales ---
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
                fila_decodificada[col] = decoded
                fila_decodificada.drop(columns=ohe_cols, inplace=True)

    fila_decodificada = fila_decodificada.squeeze()

    # --- Preparar etiquetas explicativas ---
    exp_list = exp.as_list(label=class_index)
    features_names = [f[0] for f in exp_list]
    importance = [f[1] for f in exp_list]

    etiquetas = []
    for f in features_names:
        match_col = None
        for original_col in encoders.keys():
            if f.startswith(original_col + '=') or f.startswith(original_col + '<') or f.startswith(original_col + '>'):
                match_col = original_col
                break
            # Check if OneHot encoded name appears
            encoder = encoders[original_col]
            if isinstance(encoder, OneHotEncoder):
                for ohe_col in encoder.get_feature_names_out([original_col]):
                    if f.startswith(ohe_col):
                        match_col = original_col
                        break

        if match_col and match_col in fila_decodificada:
            etiquetas.append(f"{match_col} = {fila_decodificada[match_col]}")
        else:
            etiquetas.append(f)

    # --- Gráfico ---
    plt.figure(figsize=(12, 6))
    plt.barh(etiquetas, importance, color=["red" if x < 0 else "green" for x in importance])
    plt.xlabel("Importancia")
    plt.ylabel("Características")
    plt.title(f"Explicación LIME - fila {row} (clase índice {class_index})")
    plt.yticks(fontsize=12)
    plt.axvline(0, color="black", linewidth=1)
    plt.tight_layout()
    plt.show()


def explainShapLocal(df: pd.DataFrame, model, row: int, encoders: dict, target: str, num_features: int = None, class_index=None):

    if row < 0 or row >= len(df):
        raise IndexError(f"El índice de fila {row} está fuera del rango permitido (0 a {len(df) - 1}).")

    X = df.drop(columns=target)
    X_row = X.iloc[[row]].copy()

    explainer = shap.Explainer(model.predict_proba, X, feature_names=X.columns.tolist())
    shap_values = explainer(X_row)

    # Inferir class_index si no se especifica
    if class_index is None:
        class_val = df[target].iloc[row]
        if class_val in list(model.classes_):
            class_index = list(model.classes_).index(class_val)
        else:
            raise ValueError(f"La clase real '{class_val}' no se encuentra entre las clases del modelo: {list(model.classes_)}")
    else:
        if class_index not in range(len(model.classes_)):
            raise ValueError(f"El índice de clase {class_index} no está dentro del rango válido 0 a {len(model.classes_) - 1}.")

    # Obtener valores SHAP
    if len(shap_values.shape) == 3:
        values = shap_values.values[0, :, class_index]
    else:
        values = shap_values.values[0, :]

    features = shap_values.feature_names
    importances = np.abs(values)

    if num_features is None:
        num_features = len(features)
    n_features_to_plot = min(num_features, len(features))
    sorted_idx = np.argsort(importances)[::-1][:n_features_to_plot]

    # Decodificar la fila con mapeo de columnas codificadas
    feature_label_map = {}

    for col, encoder in encoders.items():
        if isinstance(encoder, OneHotEncoder):
            ohe_cols = encoder.get_feature_names_out([col])
            if all(c in X.columns for c in ohe_cols):
                decoded_val = encoder.inverse_transform(X_row[ohe_cols])[0][0]
                for c in ohe_cols:
                    feature_label_map[c] = f"{col} = {decoded_val}"
        elif isinstance(encoder, LabelEncoder):
            if col in X_row:
                val = encoder.inverse_transform([int(X_row[col].values[0])])[0]
                feature_label_map[col] = f"{col} = {val}"
        elif isinstance(encoder, OrdinalEncoder):
            if col in X_row:
                val = encoder.inverse_transform(X_row[[col]])[0][0]
                feature_label_map[col] = f"{col} = {val}"

    # También incluir columnas numéricas no codificadas
    for col in X_row.columns:
        if col not in feature_label_map:
            feature_label_map[col] = f"{col} = {X_row[col].values[0]}"

    # Crear etiquetas finales ordenadas
    y_labels = [feature_label_map.get(features[i], features[i]) for i in sorted_idx]

    # Gráfico local
    plt.figure(figsize=(12, max(6, n_features_to_plot * 0.5)))
    plt.barh(
        y=np.array(y_labels)[::-1],
        width=values[sorted_idx][::-1],
        color=["green" if v > 0 else "red" for v in values[sorted_idx][::-1]]
    )
    plt.xlabel("Importancia SHAP")
    plt.title(f"Explicación SHAP - fila {row} (clase índice {class_index})")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def explainShapGlobal(df: pd.DataFrame, model, encoders: dict, target: str, class_index=None):
    X = df.drop(columns=target)

    explainer = shap.Explainer(model.predict_proba, X)
    shap_values = explainer(X)

    # Manejo de múltiples clases
    if class_index is None:
        class_index = 0
    elif class_index not in range(len(model.classes_)):
        raise ValueError(f"Índice de clase inválido: {class_index}. Debe estar entre 0 y {len(model.classes_) - 1}")

    if len(shap_values.shape) == 3:
        values = shap_values.values[:, :, class_index]
    else:
        values = shap_values.values

    features = shap_values.feature_names
    shap_df = pd.DataFrame(np.abs(values), columns=features)

    # Mapear columnas codificadas a columnas originales
    col_mapping = {}

    for col, encoder in encoders.items():
        if isinstance(encoder, OneHotEncoder):
            ohe_cols = encoder.get_feature_names_out([col])
            for c in ohe_cols:
                col_mapping[c] = col
        else:
            col_mapping[col] = col  # LabelEncoder / OrdinalEncoder

    for col in shap_df.columns:
        if col not in col_mapping:
            col_mapping[col] = col  # columnas numéricas o no codificadas

    # Agrupar importancias por columna original
    shap_df.columns = [col_mapping[c] for c in shap_df.columns]
    grouped = shap_df.groupby(shap_df.columns, axis=1).sum()

    # Calcular importancia media
    mean_importance = grouped.mean().sort_values(ascending=False)

    # Plot personalizado
    plt.figure(figsize=(10, max(5, 0.4 * len(mean_importance))))
    plt.barh(mean_importance.index[::-1], mean_importance.values[::-1], color="skyblue")
    plt.xlabel("Importancia SHAP media (|valor|)")
    plt.title(f"Importancia Global SHAP - Clase {class_index}")
    plt.tight_layout()
    plt.show()

