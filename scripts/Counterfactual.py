import pandas as pd
import dice_ml
from scripts.Encoding import decode

def generate_counterfactuals(model, df: pd.DataFrame, target: str, row: pd.DataFrame, encoders: dict, continuous_features=None, method="kdtree", target_value=None):
    # Obtener el valor actual de la variable objetivo en la instancia seleccionada (row)
    current_value = row[target].values[0] if target in row.columns else None
    
    # Si el valor actual ya es el objetivo deseado, no tiene sentido generar contrafactuales
    if current_value == target_value:
        raise ValueError(f"La instancia ya tiene el valor objetivo '{target_value}' en la variable '{target}'. No se puede generar un contraejemplo.")

    # Si no se proporcionan características continuas, se inicializa como lista vacía
    continuous_features = continuous_features or []
    
    # Si no se proporciona un valor objetivo, se usará el valor contrario ("opposite") por defecto
    target_value = target_value or "opposite"

    # Se elimina la columna objetivo de la instancia de entrada, ya que DiCE no la necesita para generar contrafactuales
    query_instance = row.drop(columns=[target], errors="ignore")
    
    # Inicializar los objetos de DiCE: datos y modelo
    try:
        # Detectar automáticamente las columnas categóricas por tipo 'object' o 'category'
        categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Se construye la interfaz de datos para DiCE
        d_data = dice_ml.Data(dataframe=df, continuous_features=continuous_features, outcome_name=target, categorical_features=categorical_columns)
        
        # Se construye la interfaz del modelo ya entrenado
        d_model = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")

        # Se crea el explicador DiCE con el método especificado (por defecto "kdtree")
        explainer = dice_ml.Dice(d_data, d_model, method=method)
    except Exception as e:
        # Si falla la inicialización, se lanza un error con detalle
        raise ValueError(f"Error iniciando DiCE: {str(e)}")

    # Generar los contraejemplos
    try:
        # Se generan 4 contrafactuales para la instancia dada
        cf_df = explainer.generate_counterfactuals(query_instance, total_CFs=4, desired_class=target_value).cf_examples_list[0].final_cfs_df.copy()
    except Exception as e:
        # Si ocurre un error durante la generación, se captura y lanza
        raise ValueError(f"Error generando contrafactuales: {str(e)}")

    # Decodificar la instancia original (row) y los contrafactuales generados, utilizando los encoders
    original_decoded = decode(row, encoders, target)
    cf_decoded = decode(cf_df, encoders, target)

    # Se eliminan las columnas del target para comparar solo las características
    original_features = original_decoded.drop(columns=[target], errors='ignore')

    # Identificar qué características han cambiado entre el original y los contrafactuales
    common = cf_decoded.columns.intersection(original_features.columns)  # columnas comunes
    changed = common[~cf_decoded[common].eq(original_features.iloc[0]).all()]  # columnas que han cambiado
    original_features = original_features[changed]  # conservar solo las columnas modificadas
    cf_decoded = cf_decoded[changed]  # idem para los contrafactuales

    # Mostrar resultados por consola
    print("\n[Original] Features modificadas:")
    print(original_features)

    print("\n[Contrafactuales generados]:")
    print(cf_decoded)

    priority_order = {}

    # Extraer orden de cada OrdinalEncoder si existe en los encoders
    for col, encoder in encoders.items():
        if hasattr(encoder, "categories_"):  # Verifica que sea un OrdinalEncoder entrenado
            # Extraer lista de categorías ordenadas
            categories = encoder.categories_
            if isinstance(categories, list) and len(categories) == 1:
                priority_order[col] = list(categories[0])
            elif isinstance(categories, list) and len(categories) == len(df[[col]].T):
                priority_order[col] = list(categories[df.columns.get_loc(col)])

    for i, (_, cf_row) in enumerate(cf_decoded.iterrows()):
        print(f"\n📎 Contrafactual {i+1}:")
        for col in changed:
            original_val = original_features.iloc[0][col]
            cf_val = cf_row[col]

            if original_val != cf_val:
                order = priority_order.get(col)
                if order and order.index(cf_val) >= order.index(original_val):
                    print(f"- Cambiar '{col}' de '{original_val}' a '{cf_val}' para mejorar el resultado.")
                else:
                    print(f"- Cambiar '{col}' de '{original_val}' a '{cf_val}'.")

    # Retornar el conjunto de características modificadas originales y las nuevas generadas