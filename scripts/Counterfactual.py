import pandas as pd
import dice_ml

class Counterfactual:

    def __init__(self, model, data, target, continuous_features=None, method="kdtree"):
        # Este es el método constructor de la clase.
        # --- Argumentos del Constructor ---
        # self: Referencia a la instancia de la clase que se está creando.
        # model: El modelo de aprendizaje automático para el cual se generarán
        #        explicaciones contrafactuales. Se espera que sea compatible con DiCE.
        # data: Un DataFrame de pandas que contiene el conjunto de datos utilizado para entrenar el modelo
        #       o un conjunto de datos representativo. DiCE lo usará para entender el espacio de características.
        # target: Un string con el nombre de la columna en el DataFrame 'data' que representa
        #         la variable objetivo.
        # continuous_features: Una lista (opcional, por defecto None) de nombres de columnas en 'data'
        #                      que deben ser tratadas como características continuas por DiCE.
        #                      Si es None, se inicializa como una lista vacía.
        # method: Un string (opcional, por defecto "kdtree") que especifica el método que DiCE
        #         usará para generar contrafactuales. DiCE ofrece varios métodos como
        #         "random", "genetic", "kdtree".

        # 1. Validaciones de los argumentos de entrada:
        # Comprueba si el objeto 'model' proporcionado tiene un método 'predict' y si ese método es llamable.
        # Los modelos de scikit-learn y otros compatibles con DiCE deben tener esta capacidad.
        if not hasattr(model, 'predict') or not callable(getattr(model, 'predict')):
            raise ValueError("El modelo debe tener un método 'predict' llamable.")
        
        # Comprueba si los 'data' proporcionados son una instancia de DataFrame de pandas.
        # DiCE espera los datos en este formato.
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Los datos deben ser un DataFrame de pandas.")
        
        # Comprueba si la columna 'target' especificada existe dentro del DataFrame 'data'.
        if target not in data.columns:
            raise ValueError(f"La columna objetivo '{target}' no se encuentra en los datos.")
        
        # 2. Inicialización de 'continuous_features':
        # Si 'continuous_features' no se proporciona (es None), se establece como una lista vacía.
        # Esto asegura que siempre sea una lista, lo cual es esperado por DiCE.
        continuous_features = continuous_features or []
        
        # 3. Configuración e inicialización de los componentes de DiCE:
        # Esto se hace dentro de un bloque try-except para manejar posibles errores durante la inicialización de DiCE.
        try:
            # Crea un objeto 'Model' de DiCE. Este objeto envuelve tu modelo de aprendizaje automático
            # y le indica a DiCE que es un modelo de backend "sklearn".
            dmodel = dice_ml.Model(model=model, backend="sklearn") # Asume backend sklearn; podría ser configurable

            # Crea un objeto 'Data' de DiCE. Este objeto contiene la información sobre tu conjunto de datos
            # que DiCE necesita, como el DataFrame, la lista de características continuas y
            # el nombre de la columna objetivo.
            d = dice_ml.Data(dataframe=data, continuous_features=continuous_features, outcome_name=target)
            
            # Crea la instancia principal del explicador DiCE ('Dice').
            # - 'd': El objeto Data de DiCE.
            # - 'dmodel': El objeto Model de DiCE.
            # - 'method': El método a utilizar para generar los contrafactuales.
            # El objeto explainer se almacena en 'self.exp' para ser usado por otros métodos de la clase.
            self.exp = dice_ml.Dice(d, dmodel, method=method)
        except Exception as e:
            # Si ocurre cualquier error durante la inicialización de los objetos de DiCE,
            # se captura la excepción y se lanza un ValueError más informativo.
            raise ValueError(f"Hubo un error al inicializar DiceML: {e}")
        
        # 4. Almacenar los argumentos principales como atributos de la instancia:
        # Esto permite que otros métodos de la clase puedan acceder fácilmente al modelo,
        # los datos originales y el nombre de la columna objetivo.
        self.model = model
        self.data = data
        self.target = target

    def counterfac(self, row, encoders, target_value=None):
        # Este método tiene como objetivo generar explicaciones contrafactuales
        # para una instancia (fila) de datos dada, utilizando el explicador DiCE
        # almacenado en self.exp.

        # Argumentos:
        # - 'self': Referencia a la instancia de la clase.
        # - 'row': Un DataFrame de pandas que contiene la(s) instancia(s) de consulta
        #          para las cuales se generarán contrafactuales. Aunque el nombre es 'row',
        #          DiCE puede manejar múltiples filas, pero el código posterior
        #          parece tratarlo como una sola fila (cf_row.iloc[0]).
        #          Idealmente, si es una sola fila, debería pasarse como un DataFrame de 1 fila.
        # - 'encoders': Un diccionario que contiene los codificadores ajustados que se usaron en el preprocesamiento.
        #               Se usará para decodificar los contrafactuales a un formato legible.
        # - 'target_value': El valor deseado para la columna objetivo en los contrafactuales.
        #                   Puede ser un valor específico o "opposite"
        #                   para buscar la clase contraria a la predicción original de 'row'.

        # 1. Validación de la entrada 'row':
        # Comprueba si 'row' es una instancia de DataFrame de pandas.
        if not isinstance(row, pd.DataFrame):
            raise ValueError("La fila debe ser un DataFrame de pandas.")
        if row.empty:
            raise ValueError("El DataFrame 'row' de entrada está vacío.")


        # 2. Establecer el valor objetivo deseado para los contrafactuales:
        # Si 'target_value' no se proporciona, se establece en "opposite".
        # "opposite" es una palabra clave que DiCE entiende para buscar la clase opuesta
        # a la que el modelo predice para la instancia de consulta.
        if target_value is None: # Usar 'is None' es más idiomático que '== None'
            target_value = "opposite"

        # 3. Preparar la instancia de consulta para DiCE:
        # DiCE espera que la instancia de consulta (query_instance) no contenga la columna objetivo.
        # Se crea 'cf_row' eliminando la columna objetivo ('self.target') de la 'row' original.
        # Esto asume que 'row' todavía contiene la columna objetivo. Si 'row' ya es solo
        # características, este paso podría dar un error si self.target no está en row.columns.
        if self.target in row.columns:
            cf_row_features_only = row.drop(columns=[self.target])
        else:
            # Si la columna objetivo ya no está, asumimos que 'row' ya son solo características.
            cf_row_features_only = row.copy()


        # 4. Generar explicaciones contrafactuales:
        # Se utiliza el método generate_counterfactuals del objeto explainer de DiCE (self.exp).
        # - 'cf_row_features_only': La(s) instancia(s) de consulta (solo características).
        # - 'total_CFs=4': Pide a DiCE que genere hasta 4 ejemplos contrafactuales.
        # - 'desired_class=target_value': Especifica el resultado deseado para los contrafactuales.
        # Se envuelve en un bloque try-except para manejar errores durante la generación.
        try:
            # Asegurarse de que cf_row_features_only no esté vacío
            if cf_row_features_only.empty:
                raise ValueError("La instancia de consulta (sin la columna objetivo) está vacía.")

            dice_exp = self.exp.generate_counterfactuals(
                cf_row_features_only,
                total_CFs=4,
                desired_class=target_value
            )
            if dice_exp is None or not dice_exp.cf_examples_list:
                print("DiCE no pudo generar contrafactuales para la instancia dada con los parámetros actuales.")
                return row, pd.DataFrame() # Devolver la fila original y un DataFrame vacío
        except Exception as e:
            raise ValueError(f"Error al generar contrafactuales con DiCE: {e}")
        
        # 5. Extraer los DataFrames contrafactuales:
        # DiCE devuelve los resultados en una estructura anidada.
        # 'dice_exp.cf_examples_list[0]' accede a los resultados para la primera instancia de consulta.
        # '.final_cfs_df' es un DataFrame que contiene los ejemplos contrafactuales generados.
        # Este DataFrame puede tener características en su formato codificado,
        # dependiendo de cómo se haya configurado DiCE y el modelo.
        if not dice_exp.cf_examples_list[0].final_cfs_df.empty:
            cf_data = dice_exp.cf_examples_list[0].final_cfs_df.copy() # Hacer una copia para evitar SettingWithCopyWarning
        else:
            print("DiCE generó una lista de ejemplos, pero no hay DataFrames contrafactuales finales.")
            return row, pd.DataFrame()


        # 6. Decodificar características:
        # Este método tomaría un DataFrame y el diccionario de 'encoders' y devolvería
        # un DataFrame con las características categóricas decodificadas a sus valores originales.
        # Esto es crucial para la interpretabilidad.
        # Se decodifican tanto los contrafactuales ('cf_data') como la fila original ('row')
        # para asegurar que la comparación posterior de columnas sin cambios se haga sobre datos comparables.
        # Es importante que el método __decode maneje correctamente los nombres de las columnas.
        if hasattr(self, '_LimeExplainerPipeline__decode'): # Ejemplo de nombre de método privado
             cf_data_decoded = self._LimeExplainerPipeline__decode(cf_data, encoders)
             row_decoded = self._LimeExplainerPipeline__decode(row, encoders) # Decodificar la fila original completa (incluyendo el target)
        elif hasattr(self, '__decode'): # Nombre genérico
             cf_data_decoded = self.__decode(cf_data, encoders)
             row_decoded = self.__decode(row, encoders)
        else:
            print("Advertencia: Método de decodificación no encontrado. Los contrafactuales podrían estar codificados.")
            cf_data_decoded = cf_data # Usar datos como están si no hay decodificación
            row_decoded = row     # Usar fila original como está

        # La fila original para comparación debería ser solo las características y decodificada
        # cf_row_features_only_decoded = self.__decode(cf_row_features_only, encoders)
        # Usaremos row_decoded (que es la fila original completa decodificada) y extraeremos características
        # si es necesario, o directamente row_decoded si __decode solo devuelve características.
        # Para la comparación, necesitamos que row_decoded tenga las mismas columnas de características que cf_data_decoded.
        
        # Asegurarse de que row_decoded para comparación solo tenga las características que están en cf_data_decoded
        # Si row_decoded incluye el target, hay que quitarlo para la comparación de .eq()
        row_features_decoded_for_comparison = row_decoded.copy()
        if self.target in row_features_decoded_for_comparison.columns:
            row_features_decoded_for_comparison = row_features_decoded_for_comparison.drop(columns=[self.target])


        # 7. Identificar y eliminar columnas que no cambian:
        # El objetivo es mostrar solo las características que DiCE modificó para lograr el contrafactual.
        # - cf_data_decoded.eq(row_features_decoded_for_comparison.iloc[0]): Compara cada celda del DataFrame de contrafactuales
        #   con la celda correspondiente en la primera fila de la instancia original decodificada.
        #   Esto devuelve un DataFrame de booleanos (True si son iguales, False si no).
        # - .all(): Aplicado a este DataFrame booleano, comprueba si todos los valores en cada *columna* son True.
        #   Esto resulta en una Serie booleana donde el índice son los nombres de las columnas, y el valor es True
        #   si la columna no cambió en NINGUNO de los contrafactuales generados con respecto a la fila original.
        # - cf_data_decoded.columns[...]: Selecciona los nombres de las columnas que no cambiaron.
        
        # Asegurarse de que las columnas para la comparación .eq() sean las mismas
        common_features = cf_data_decoded.columns.intersection(row_features_decoded_for_comparison.columns)
        if common_features.empty:
            print("Advertencia: No hay características comunes entre los contrafactuales y la fila original después de decodificar.")
            unchanged_columns = pd.Index([])
        else:
            # Comparar solo en características comunes
            comparison_df = cf_data_decoded[common_features].eq(row_features_decoded_for_comparison[common_features].iloc[0])
            unchanged_columns = common_features[comparison_df.all()]

        # Elimina las columnas sin cambios de los DataFrames de contrafactuales y de la fila original (versión decodificada).
        # inplace=True modifica los DataFrames directamente.
        cf_data_to_show = cf_data_decoded.drop(columns=unchanged_columns, errors='ignore')
        row_to_show_features_only = row_features_decoded_for_comparison.drop(columns=unchanged_columns, errors='ignore')


        # 8. Imprimir resultados:
        # Muestra la fila original (solo las características que cambiaron o son relevantes)
        # y los contrafactuales generados (también solo con las características que cambiaron).
        print("Fila original (solo características relevantes/modificadas):")
        print(row_to_show_features_only)
        print("\nContrafactuales generados (solo características relevantes/modificadas):")
        print(cf_data_to_show)
        
        # 9. Devolver los resultados:
        # Devuelve la fila original (decodificada, solo características relevantes) y
        # el DataFrame de contrafactuales (decodificados, solo características relevantes).
        return row_to_show_features_only, cf_data_to_show