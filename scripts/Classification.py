import pandas as pd
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class Classification:

    def __init__(self, data, target, encoders, test_size=0.2):
        # Este es el método constructor de la clase. Se llama automáticamente
        # cuando creas una nueva instancia (objeto) de esta clase. Los argumentos que recibe son:
        # - 'self': Es una referencia a la instancia de la clase que se está creando.
        #           Permite acceder a los atributos y métodos de la instancia.
        # - 'data': Se espera que sea un DataFrame de pandas que contiene el conjunto de datos completo
        #           (características y la columna objetivo).
        # - 'target': Un string con el nombre de la columna en 'data' que se usará como
        #             la variable objetivo (la que se quiere predecir).
        # - 'encoders': Un diccionario que se espera contenga los codificadores ajustados
        #               (por ejemplo, el resultado del método 'encode' que vimos antes).
        #               Esto es útil si necesitas decodificar o aplicar transformaciones inversas.
        # - 'test_size': Un valor flotante (opcional, con valor predeterminado de 0.2) que determina
        #                la proporción del conjunto de datos que se reservará para el conjunto de prueba.
        #                0.2 significa que el 20% de los datos serán para prueba y el 80% para entrenamiento.

        # 1. Separar las características (X) de la variable objetivo (Y):
        # 'Y' contendrá la Serie de pandas correspondiente a la columna objetivo.
        Y = data[target]
        # 'X' contendrá un nuevo DataFrame con todas las columnas de 'data' excepto la columna objetivo.
        # 'axis=1' indica que 'target' es el nombre de una columna a eliminar (no una fila).
        X = data.drop(target, axis=1)

        # 2. Dividir los datos en conjuntos de entrenamiento y prueba:
        # `train_test_split` es una función de scikit-learn que divide los datos aleatoriamente.
        # - X, Y: Son las características y la variable objetivo a dividir.
        # - test_size: Proporción del conjunto de datos a incluir en la división de prueba.
        # - random_state=42: Fija la semilla para la aleatoriedad. Esto asegura que la división
        #                    sea la misma cada vez que se ejecute el código, lo cual es crucial
        #                    para la reproducibilidad de los resultados.
        # La función devuelve cuatro conjuntos de datos:
        # - self.xtrain: Características para el conjunto de entrenamiento.
        # - self.xtest: Características para el conjunto de prueba.
        # - self.ytrain: Variable objetivo para el conjunto de entrenamiento.
        # - self.ytest: Variable objetivo para el conjunto de prueba.
        # Estos se guardan como atributos de la instancia para que otros métodos de la clase puedan usarlos.
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(X, Y, test_size=test_size, random_state=42)

        # 3. Almacenar los codificadores:
        # Guarda el diccionario de 'encoders' (que se pasó como argumento) como un atributo de la instancia.
        self.encoders = encoders
        
        # 4. Definir una lista de modelos de clasificación:
        # 'self.list_models' es un diccionario donde las claves son nombres descriptivos
        # de los modelos y los valores son instancias de los clasificadores de scikit-learn,
        # XGBoost y CatBoost. Este diccionario facilita la iteración sobre diferentes modelos para entrenarlos y evaluarlos.
        self.list_models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(), # KNN no tiene random_state en su constructor principal
            'SVC': SVC(probability=True, random_state=42),
            'Naive Bayes': GaussianNB(), # GaussianNB no tiene random_state
            'MLP Classifier': MLPClassifier(max_iter=1000, random_state=42),
            'XGBoost': XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42), # use_label_encoder=False para evitar warnings
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
        }

    def trainEvaluate(self):
        # Este método se encarga de entrenar y evaluar una lista de modelos de clasificación.

        # Inicializa una lista vacía llamada 'records'.
        # Esta lista se usará para almacenar un diccionario por cada modelo,
        # donde cada diccionario contendrá el nombre del modelo y sus métricas de rendimiento.
        records = []

        # Itera sobre cada par (nombre, modelo) en el diccionario 'self.list_models'.
        # 'name' será el string con el nombre del modelo (ej: 'Decision Tree').
        # 'model' será la instancia del clasificador (ej: DecisionTreeClassifier()).
        for name, model in self.list_models.items():
            try:
                # Intenta entrenar el modelo actual.
                # El método .fit() entrena el modelo utilizando las características de entrenamiento (self.xtrain)
                # y las etiquetas/objetivos de entrenamiento (self.ytrain).
                model.fit(self.xtrain, self.ytrain)
                
                # Una vez que el modelo está entrenado, se utiliza para hacer predicciones
                # sobre el conjunto de datos de prueba (self.xtest).
                # Las predicciones resultantes se almacenan en la variable 'ypred'.
                ypred = model.predict(self.xtest)
                
                # Calcula varias métricas de evaluación para comparar las predicciones ('ypred')
                # con los valores reales del conjunto de prueba ('self.ytest').

                # Accuracy (Exactitud): Proporción de predicciones correctas sobre el total.
                accuracy = accuracy_score(self.ytest, ypred)
                
                # Precision (Precisión): Mide la proporción de identificaciones positivas
                #                        (ej: clase 1) que fueron realmente correctas.
                # average='weighted': Calcula la precisión para cada clase y luego un promedio
                #                     ponderado por el soporte de cada clase (número de instancias verdaderas).
                #                     Es útil para clases desbalanceadas.
                # zero_division=1: Si no hay predicciones positivas para una clase (lo que resultaría
                #                  en una división por cero), la precisión para esa clase se establece en 1.0.
                #                  Esto evita warnings. Alternativamente, podría ser 0.
                precision = precision_score(self.ytest, ypred, average='weighted', zero_division=1)
                
                # Recall (Sensibilidad o Exhaustividad): Mide la proporción de positivos reales
                #                                       que fueron correctamente identificados por el modelo.
                # average='weighted': Similar a la precisión, para el manejo de múltiples clases.
                # zero_division=0: Si no hay instancias reales de una clase, el recall se establece en 0.
                recall = recall_score(self.ytest, ypred, average='weighted', zero_division=0)
                
                # F1-Score: Es la media armónica de la precisión y el recall. Proporciona un balance
                #           entre ambas métricas. Es especialmente útil si hay un desbalance de clases.
                # average='weighted': Similar a la precisión y recall.
                f1 = f1_score(self.ytest, ypred, average='weighted', zero_division=0)
                
                # Crea un diccionario para almacenar las métricas del modelo actual.
                record = {
                    'Model': name,        # Nombre del modelo
                    'Accuracy': accuracy, # Valor de exactitud
                    'Precision': precision, # Valor de precisión
                    'Recall': recall,     # Valor de recall
                    'F1-Score': f1        # Valor de F1-score
                }
                # Añade el diccionario de este modelo a la lista 'records'.
                records.append(record)
            
            except Exception as e:
                # Si ocurre cualquier error durante el entrenamiento (.fit()) o la predicción (.predict())
                # para el modelo actual, este bloque 'except' se ejecutará.
                print(f"Error al entrenar o predecir con el modelo {name}: {e}")
                # Se crea un registro con valores None para las métricas de este modelo fallido.
                record = {
                    'Model': name,
                    'Accuracy': None,
                    'Precision': None,
                    'Recall': None,
                    'F1-Score': None
                }
                # Se añade este registro de error a la lista 'records'.
                records.append(record)
        
        # Después de iterar sobre todos los modelos:
        # Convierte la lista de diccionarios 'records' en un DataFrame de pandas.
        # .sort_values(by='F1-Score', ascending=False): Ordena el DataFrame resultante
        #                                                basándose en la columna 'F1-Score'
        #                                                en orden descendente (los modelos con
        #                                                mayor F1-Score aparecerán primero).
        # .reset_index(drop=True): Reinicia el índice del DataFrame (para que vaya de 0, 1, 2...)
        #                          y 'drop=True' evita que el antiguo índice se añada como
        #                          una nueva columna en el DataFrame.
        return pd.DataFrame(records).sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
    
    def train(self, models_selected):
        # Este método tiene como objetivo entrenar un ensamblador de modelos llamado
        # VotingClassifier utilizando una selección de los modelos predefinidos
        # en self.list_models. Luego evalúa este VotingClassifier.

        # Argumentos:
        # - 'self': Referencia a la instancia de la clase.
        # - 'models_selected': Puede ser un string con el nombre de un solo modelo
        #                      o una lista de strings con los nombres de los modelos
        #                      que se incluirán en el VotingClassifier. Estos nombres
        #                      deben corresponder a las claves en el diccionario self.list_models.

        # 1. Normalizar la entrada 'models_selected':
        # Si 'models_selected' es un solo string,
        # se convierte en una lista que contiene ese string.
        # Esto permite que el resto del código maneje 'models_selected' siempre como una lista.
        if isinstance(models_selected, str):
            models_selected = [models_selected]
        
        # 2. Preparar la lista de estimadores para VotingClassifier:
        # 'estimators' será una lista de tuplas, donde cada tupla contiene el nombre del modelo y la instancia del modelo.
        # Esto es el formato que espera el parámetro 'estimators' de VotingClassifier.
        estimators = []

        # Itera sobre cada nombre de modelo en la lista 'models_selected'.
        for name in models_selected:
            # Comprueba si el nombre del modelo actual existe como clave en el diccionario 'self.list_models'.
            if name in self.list_models:
                # Si el modelo existe, obtiene la instancia del modelo del diccionario.
                model = self.list_models[name]
                # Añade una tupla (nombre_del_modelo, instancia_del_modelo) a la lista 'estimators'.
                estimators.append((name, model))
            else:
                # Si el nombre del modelo no se encuentra en 'self.list_models',
                # imprime una advertencia indicando que ese modelo no está disponible.
                print(f"Advertencia: El modelo '{name}' no se encuentra en la lista de modelos disponibles.")

        # 3. Validar que se haya seleccionado al menos un modelo válido:
        # Si la lista 'estimators' está vacía (lo que significa que ninguno de los nombres
        # en 'models_selected' correspondía a un modelo válido en 'self.list_models'),
        # se lanza un ValueError porque VotingClassifier necesita al menos un estimador.
        if not estimators:
            raise ValueError("No se ha seleccionado ningún modelo válido para el VotingClassifier.")

        # 4. Crear y entrenar el VotingClassifier:
        # Se crea una instancia de VotingClassifier.
        # - 'estimators=estimators': La lista de tuplas (nombre, modelo) que se usarán en el ensamble.
        # - 'voting='soft'': Especifica el tipo de votación.
        #   - 'soft': Promedia las probabilidades predichas por cada clasificador individual
        #             (si los clasificadores pueden predecir probabilidades, como SVC(probability=True)).
        #             La clase con la mayor probabilidad promedio es la predicción final.
        #             Generalmente funciona mejor si los clasificadores están bien calibrados.
        #   - 'hard': Votación por mayoría. La clase predicha es la que recibe más votos
        #             de los clasificadores individuales.
        # Se almacena el VotingClassifier entrenado en el atributo 'self.model' de la instancia.
        # Esto podría sobrescribir un 'self.model' anterior si existiera.
        self.model = VotingClassifier(estimators=estimators, voting='soft')
        
        # Se entrena el VotingClassifier usando los datos de entrenamiento (self.xtrain, self.ytrain).
        # El VotingClassifier internamente entrenará cada uno de los modelos que se le pasaron.
        self.model.fit(self.xtrain, self.ytrain)
        
        # 5. Realizar predicciones con el VotingClassifier entrenado:
        # Se utilizan los datos de prueba (self.xtest) para hacer predicciones.
        ypred = self.model.predict(self.xtest)
        
        # 6. Calcular y almacenar las métricas de rendimiento del VotingClassifier:
        # Se crea un diccionario 'record' para guardar las métricas, similar a como
        # se hizo en el método 'trainEvaluate' para modelos individuales.
        record = {
            'Model': "VotingClassifier", # Nombre para identificar este resultado
            'Accuracy': accuracy_score(self.ytest, ypred),
            'Precision': precision_score(self.ytest, ypred, average='weighted', zero_division=1),
            'Recall': recall_score(self.ytest, ypred, average='weighted', zero_division=0), # Ajustado zero_division
            'F1-Score': f1_score(self.ytest, ypred, average='weighted', zero_division=0)   # Ajustado zero_division
        }

        # 7. Devolver las métricas:
        # Se convierte el diccionario 'record' en un DataFrame de pandas de una sola fila.
        # Esto permite un formato de salida consistente con el método 'trainEvaluate'
        # si se quisieran combinar o comparar resultados.
        return pd.DataFrame([record])

    def predict(self, row):
        # Este método tiene como objetivo tomar una o más filas de datos de entrada,
        # utilizar un modelo previamente entrenado para hacer predicciones,
        # y devolver la entrada original junto con la predicción y la probabilidad de esa predicción.

        # Argumentos:
        # - 'self': Referencia a la instancia de la clase.
        # - 'row': Los datos de entrada para los cuales se quiere hacer una predicción.
        #          Puede ser una Serie de pandas (para una sola instancia/fila) o
        #          un DataFrame de pandas (para una o múltiples instancias/filas).

        # 1. Verificar si el modelo ha sido entrenado:
        # `hasattr(self, 'model')` comprueba si el atributo 'model' existe en la instancia.
        # `getattr(self, 'model')` obtiene el valor del atributo 'model'.
        # La condición completa verifica si el atributo 'model' no existe O si existe pero es None.
        # Si el modelo no está entrenado (o no existe), se lanza un ValueError.
        # Una forma más concisa podría ser `if getattr(self, 'model', None) is None:`.
        if not hasattr(self, 'model') or getattr(self, 'model') is None: # Corrección: `or` en lugar de `and` para la lógica deseada
                                                                        # O, más simple: if getattr(self, 'model', None) is None:
            raise ValueError("Se debe entrenar antes un modelo con el método train().")

        # 2. Asegurar que la entrada 'row' sea un DataFrame:
        # Los modelos de scikit-learn generalmente esperan una entrada 2D.
        # Si 'row' es una Serie de pandas, se convierte a un DataFrame de una sola fila.
        # - `row.to_frame()`: Convierte la Serie a un DataFrame de una sola columna.
        # - `.T`: Transpone el DataFrame, convirtiendo la columna en una fila.
        if isinstance(row, pd.Series):
            row = row.to_frame().T
        # Si 'row' ya es un DataFrame, no se modifica.

        # 3. Realizar predicciones:
        # Se utiliza el modelo entrenado ('self.model') para predecir la clase.
        # 'ypred' será un array con la(s) clase(s) predicha(s).
        ypred = self.model.predict(row)
        
        # Se utiliza el modelo entrenado para predecir las probabilidades de cada clase.
        # 'yprob' será un array 2D donde cada fila corresponde a una instancia de entrada,
        # y cada columna corresponde a la probabilidad predicha para una clase.
        # Por ejemplo, para clasificación binaria, yprob podría ser [[0.1, 0.9], [0.8, 0.2], ...],
        # donde la primera columna es la probabilidad de la clase 0 y la segunda de la clase 1.
        yprob = self.model.predict_proba(row)
        
        # 4. Preparar la salida:
        # Se crea un nuevo DataFrame 'row_pred' concatenando la(s) fila(s) de entrada originales ('row')
        # con las predicciones de clase ('ypred').
        # - `pd.DataFrame(ypred, columns=[self.ytrain.name])`: Crea un DataFrame a partir de las predicciones.
        #   Se asume que 'self.ytrain.name' contiene el nombre de la columna objetivo original, y este nombre se usa para la nueva columna de predicciones.
        #   El índice de este nuevo DataFrame se alineará con el de 'row' si 'row' tiene un índice estándar.
        #   Para que la concatenación funcione bien, es mejor resetear el índice de 'row' si no es estándar.
        #   o asegurarse de que los índices sean compatibles.
        
        # Asegurar que los índices se alineen para la concatenación
        current_row_index = row.index
        row_pred = pd.concat(
            [row.reset_index(drop=True), 
            pd.DataFrame(ypred, columns=[getattr(self.ytrain, 'name', 'Prediction')]).reset_index(drop=True)
            ], axis=1
        )
        row_pred.index = current_row_index # Restaurar el índice original si es necesario y deseado

        # 5. Extraer la probabilidad de la clase predicha:
        # Para cada predicción en 'ypred', se quiere la probabilidad asociada a esa clase predicha
        # del array 'yprob'.
        # - `yprob[i]`: Son las probabilidades para todas las clases de la i-ésima instancia.
        # - `ypred[i]`: Es la clase predicha para la i-ésima instancia.
        # - `yprob[i][ypred[i]]`: Selecciona la probabilidad de la clase específica que fue predicha.
        prob_values = [yprob[i][ypred[i]] for i in range(len(ypred))]
        
        # Convertir las probabilidades a porcentajes (multiplicando por 100).
        prob_values_percent = [p * 100 for p in prob_values]

        # 6. Devolver los resultados:
        # - 'row_pred': El DataFrame original con la columna de predicción añadida.
        # - `pd.Series(prob_values_percent).round(2)`: Una Serie de pandas con las probabilidades
        #   de la clase predicha, redondeadas a 2 decimales.
        #   Se le podría asignar el índice de 'row' también si se devuelve una Serie.
        
        # Crear la Serie de probabilidades con el mismo índice que la entrada original
        prob_series = pd.Series(prob_values_percent, index=current_row_index, name="Predicted_Probability_%").round(2)

        return row_pred, prob_series

    def explain(self, row, num_features = None):

        if getattr(self, 'model') is None:
            raise ValueError("Se debe entrenar antes un modelo con el método train().")

        if row < 0 or row >= len(self.xtest):
            raise IndexError(f"El índice de fila {row} está fuera del rango permitido (0 a {len(self.xtest) - 1}).")

        self.data_row = self.xtest.iloc[[row]].copy()
        self.oe_encoders = {}

        for col, encoder in self.encoders.items():
            if isinstance(encoder, OneHotEncoder):
                ohe_cols = encoder.get_feature_names_out([col])
                self.data_row[col] = encoder.inverse_transform(self.data_row[ohe_cols])
                self.data_row = self.data_row.drop(columns=ohe_cols)
                self.oe_encoders[col] = OrdinalEncoder()
                self.data_row[col] = self.oe_encoders[col].fit_transform(self.data_row[[col]])
        
        categorical_feature_names = [col for col in self.encoders.keys() if col != self.ytrain.name]
        categorical_feature_indices = [self.data_row.columns.get_loc(col) for col in categorical_feature_names]

        categorical_names = {}
        for col in categorical_feature_names:
            encoder = self.encoders[col]
            if hasattr(encoder, "categories_"):
                # Para OneHotEncoder o OrdinalEncoder
                categories = encoder.categories_[0].tolist()
            elif hasattr(encoder, "classes_"):
                # Para LabelEncoder
                categories = encoder.classes_.tolist()
            else:
                categories = []
            
            categorical_names[self.data_row.columns.get_loc(col)] = categories

        self.xdata = self.xtrain.copy()
        self.oe_encoders = {}

        for col, encoder in self.encoders.items():
            if isinstance(encoder, OneHotEncoder):
                ohe_cols = encoder.get_feature_names_out([col])
                self.xdata[col] = encoder.inverse_transform(self.xdata[ohe_cols])
                self.xdata.drop(columns=ohe_cols, inplace = True)
                self.oe_encoders[col] = OrdinalEncoder()
                self.xdata[col] = self.oe_encoders[col].fit_transform(self.xdata[[col]])

        self.explainer = lime.lime_tabular.LimeTabularExplainer(training_data = self.xdata.values,
                                                    feature_names = self.data_row.columns.tolist(),
                                                    class_names = self.ytrain.unique().tolist(),
                                                    categorical_features=categorical_feature_indices,
                                                    categorical_names=categorical_names,
                                                    mode="classification",
                                                    random_state=42)

        if num_features is None:
            num_features = len(self.xdata.columns.tolist())

        data = self.data_row.iloc[0]

        exp = self.explainer.explain_instance(data_row=data, predict_fn=self.__lime_predict_fn, num_features=num_features)
        exp_list = exp.as_list()
        features_names = [f[0] for f in exp_list]
        importance = [f[1] for f in exp_list]

        plt.figure(figsize=(12, 6))
        plt.barh(features_names, importance, color=["red" if x < 0 else "green" for x in importance])
        plt.xlabel("Importancia")
        plt.ylabel("Características")
        plt.title("Explicación de LIME")
        plt.yticks(fontsize=12)
        plt.axvline(0, color="black", linewidth=1)
        plt.tight_layout()
        plt.show()

    def __lime_predict_fn(self, x_ordinal_encoded_samples_np):
        # Convertir el array NumPy a DataFrame usando los nombres de columna originales
        data = pd.DataFrame(x_ordinal_encoded_samples_np, columns=self.data_row.columns.tolist())

        for col, encoder in self.encoders.items():
            if isinstance(encoder, OneHotEncoder):
                data[col] = self.oe_encoders[col].inverse_transform(data[[col]]).ravel()
                transformed = encoder.transform(data[[col]])
                columns = encoder.get_feature_names_out([col])
                df_ohe = pd.DataFrame(transformed, columns=columns, index=data.index)
                data = data.drop(columns=[col]).join(df_ohe)

        # Crear una copia para invertir la codificación ordinal
        probabilities = self.model.predict_proba(data.copy())
        return probabilities

