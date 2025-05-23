import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from imblearn.over_sampling import SMOTENC, SMOTE, SMOTEN
import nltk
try:
    from nltk.corpus import stopwords
    stopwords.words('spanish')  # o 'english', según idioma
except LookupError:
    nltk.download('stopwords')

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class Preprocessing:

    def __init__(self, link, encoding='utf-8', delimiter=';'):
        # - 'self': Es una referencia a la instancia de la clase que se está creando.
        # - 'link': Se espera que sea una cadena de texto (string) que contenga la ruta
        #           o URL al archivo CSV que se va a leer.
        # - 'encoding': El tipo de codificación del archivo CSV (ej: 'utf-8', 'latin1'). Por defecto es 'utf-8'
        # - 'delimiter': El carácter que separa los valores en el archivo CSV (ej: ',', ';', '\t'). Por defecto es ';'

        # Crea un diccionario llamado 'read_kwargs' para almacenar los argumentos
        # que se pasarán a la función pd.read_csv.
        # Inicialmente, contiene 'encoding' y 'delimiter' con los valores recibidos.
        read_kwargs = {'encoding': encoding, 'delimiter': delimiter}

        # Esta línea filtra el diccionario 'read_kwargs'.
        # Crea un nuevo diccionario que solo incluye las claves (k) y valores (v)
        # del 'read_kwargs' original si el valor (v) no es None.
        # Esto es útil para que, si por ejemplo 'encoding' o 'delimiter' no se especifican
        # (son None), no se pasen como argumentos explícitos con valor None a pd.read_csv,
        # permitiendo que pd.read_csv use sus propios valores predeterminados en esos casos.
        read_kwargs = {k: v for k, v in read_kwargs.items() if v is not None}

        # Lee el archivo CSV utilizando la función read_csv de la biblioteca pandas.
        # - 'link': Es el primer argumento, la ruta o URL del archivo.
        # - '**read_kwargs': Desempaqueta el diccionario 'read_kwargs'.
        self.data = pd.read_csv(link, **read_kwargs)
        
        # Este bucle itera sobre todas las columnas del DataFrame 'self.data'
        # que son de tipo 'object' o 'string'. Estos tipos de datos suelen usarse
        # en pandas para almacenar texto.
        for col in self.data.select_dtypes(include=['object', 'string']):
            # Para cada una de estas columnas ('col'):
            # 1. .astype(str): Se asegura de que todos los valores en la columna sean tratados como strings.
            #    Esto es útil si la columna tiene tipos mixtos o valores NaN que podrían causar problemas
            #    con las operaciones de string.
            # 2. .str.strip(): Elimina los espacios en blanco al principio y al final de cada string
            #    en la columna.
            # 3. .str.lower(): Convierte todos los caracteres de cada string en la columna a minúsculas.
            self.data[col] = self.data[col].astype(str).str.strip().str.lower()

        for col in self.data.columns:
            # Para cada columna ('col'):
            # Intenta convertir los valores de la columna a un tipo numérico (float o int)
            # usando pd.to_numeric.
            # - 'errors='ignore'': Si un valor en la columna no puede convertirse a número,
            #   esta opción hace que pd.to_numeric no lance un error, sino que deje el valor
            #   original sin convertir. Las columnas que sí puedan convertirse completamente
            #   a numéricas cambiarán su tipo de dato (dtype); las que no, permanecerán
            #   como tipo 'object' o el tipo que tenían.
            self.data[col] = pd.to_numeric(self.data[col], errors='ignore')

    def add(self, column_name, values):
        # - 'self': La referencia a la instancia de la clase.
        # - 'column_name': Un string que será el nombre de la nueva columna a añadir.
        # - 'values': Los datos que se asignarán a la nueva columna. Se espera que sea
        #             un iterable con la misma longitud que el número de filas del DataFrame.

        # Primera comprobación de validación para el argumento 'values'.
        # isinstance(values, str): Verifica si 'values' es un único string.
        # not hasattr(values, '__len__'): Verifica si 'values' no tiene el atributo '__len__'.
        #                                  La mayoría de los iterables (listas, Series, etc.) tienen __len__.
        #                                  Un solo número, por ejemplo, no lo tendría.
        # Si 'values' es un string O no es un iterable con longitud definida, se lanza un TypeError.
        # Esto es para evitar que se intente asignar un solo string como si fuera una columna completa
        # o un objeto no iterable que no podría formar una columna.
        if isinstance(values, str) or not hasattr(values, '__len__'):
            raise TypeError("El argumento 'values' debe ser una lista, Serie u otro iterable, no un string.")

        # Segunda comprobación de validación: la longitud de 'values' debe coincidir
        # con el número de filas del DataFrame 'self.data'.
        # len(self.data) devuelve el número de filas del DataFrame.
        if len(values) == len(self.data):
            # Si las longitudes coinciden, se crea una nueva columna en 'self.data'
            # con el nombre 'column_name' y se le asignan los 'values'.
            # Si la columna ya existe, se sobrescribirá.
            self.data[column_name] = values
        else:
            # Si las longitudes no coinciden, se lanza un ValueError porque pandas
            # requiere que la longitud de los datos de la nueva columna sea igual
            # al número de filas existentes en el DataFrame.
            raise ValueError("La longitud del array no coincide con el número de filas del DataFrame.")
        
    def delete(self, columns):
        # - 'self': La referencia a la instancia de la clase.
        # - 'columns': Puede ser un único nombre de columna (string) o una lista de nombres de columnas
        #              que se desean eliminar del DataFrame self.data.

        # 1. Asegurar que 'columns' sea una lista:
        # Si el argumento 'columns' es un único string, esta línea lo convierte en una lista que contiene ese string.
        # Esto permite que el resto del código trate 'columns' siempre como una lista,
        # simplificando el manejo tanto para una como para múltiples columnas a eliminar.
        if isinstance(columns, str):
            columns = [columns]
        # Si 'columns' ya es una lista, no se modifica.

        # 2. Validar que todas las columnas a eliminar existan en el DataFrame:
        # Se crea una lista llamada 'missing' usando una comprensión de listas.
        # Esta lista contendrá los nombres de las columnas que están en el argumento 'columns'
        # pero NO se encuentran en las columnas actuales del DataFrame.
        missing = [col for col in columns if col not in self.data.columns]

        # Si la lista 'missing' no está vacía, significa que se intentó eliminar
        # al menos una columna que no existe.
        if missing:
            # En este caso, se lanza un error de tipo KeyError.
            # El mensaje de error es informativo, indicando cuáles columnas específicas
            # no se encontraron, lo que ayuda al usuario a depurar.
            raise KeyError(f"Las siguientes columnas no se encontraron en el DataFrame: {missing}")
        
        # 3. Eliminar las columnas
        # Si todas las validaciones anteriores pasan, se procede a eliminar las columnas del DataFrame.
        # - 'columns': Es la lista de nombres de columnas a eliminar.
        # - 'axis=1': Indica que estamos eliminando columnas.
        # - 'inplace=True': Modifica el DataFrame 'self.data' directamente.
        try:
            self.data.drop(columns=columns, axis=1, inplace=True)
            print(f"Columnas {columns} eliminadas exitosamente.")
        except Exception as e:
            # Capturar cualquier otro error inesperado durante la eliminación
            print(f"Ocurrió un error al intentar eliminar las columnas: {e}")
            # Podrías optar por relanzar el error o manejarlo de otra forma
            raise

    def replace(self, column1, column1_value, column2, column2_value, new_column2_value, case_sensitive=False):
        # - 'self': Referencia a la instancia de la clase.
        # - 'column1': Nombre de la primera columna para la condición.
        # - 'column1_value': Valor a buscar en 'column1'.
        # - 'column2': Nombre de la segunda columna, tanto para la condición como para la actualización.
        # - 'column2_value': Valor a buscar en 'column2' como parte de la condición.
        # - 'new_column2_value': El nuevo valor que se asignará a 'column2' si se cumple la condición.
        # - 'case_sensitive': Un booleano (opcional, por defecto False) que determina si las
        #                     comparaciones de 'column1_value' y 'column2_value' deben ser
        #                     sensibles a mayúsculas y minúsculas.

        # 1. Validación de existencia de columnas:
        # Comprueba si los nombres de columna proporcionados ('column1' y 'column2')
        # existen realmente en el DataFrame 'self.data'.
        if column1 not in self.data.columns or column2 not in self.data.columns:
            # Si alguna de las columnas no existe, lanza un error de tipo KeyError
            # con un mensaje informativo.
            raise KeyError(f"Las columnas {column1} o {column2} no existen en el DataFrame.")

        # 2. Manejo de sensibilidad a mayúsculas y minúsculas para los valores de condición:
        # Si 'case_sensitive' es False (que es el valor por defecto),
        # convierte los valores de búsqueda 'column1_value' y 'column2_value' a minúsculas.
        # Esto asegura que la comparación posterior se haga sin distinguir mayúsculas/minúsculas.
        if not case_sensitive:
            # Solo intenta convertir a minúsculas si los valores son strings.
            if isinstance(column1_value, str):
                column1_value = column1_value.lower()
            if isinstance(column2_value, str):
                column2_value = column2_value.lower()
            # Nota: new_column2_value se asignará tal cual, sin conversión de mayúsculas/minúsculas
            # basada en case_sensitive. Si se desea, se podría añadir lógica similar aquí.

        # 3. Construcción de la condición para la selección de filas:
        # Se utiliza el accesor .loc de pandas para seleccionar filas y columnas.
        # La primera parte de .loc (antes de la coma) es la condición para las filas.

        # Condición para la sustitución:
        # (self.data[column1].astype(str) == str(column1_value))
        # - self.data[column1]: Selecciona la columna 'column1'.
        # - astype(str): Trata el valor de esa columna como string
        # Si no es sensible a mayúsculas, comparamos con:
        # - str.lower(): convierte el string en minúscula
        if not case_sensitive:
            condition1 = (self.data[column1].astype(str).str.lower() == column1_value)
            condition2 = (self.data[column2].astype(str).str.lower() == column2_value)
        else:
            condition1 = (self.data[column1].astype(str) == str(column1_value))
            condition2 = (self.data[column2].astype(str) == str(column2_value))
        
        # '&': El operador AND lógico. Combina las dos condiciones. Una fila se seleccionará
        #      solo si ambas condiciones son verdaderas para esa fila.
        combined_condition = condition1 & condition2

        # 4. Asignación del nuevo valor:
        # self.data.loc[combined_condition, column2] = new_column2_value
        # - combined_condition: Selecciona las filas que cumplen ambas condiciones.
        # - column2: Especifica que la actualización se hará en la columna 'column2' para esas filas.
        # - = new_column2_value: Asigna 'new_column2_value' a las celdas seleccionadas.
        self.data.loc[combined_condition, column2] = new_column2_value
    
    def standarize (self, column_name, link):
        # Este método toma tres argumentos:
        # - 'self': La referencia a la instancia de la clase.
        # - 'column_name': El nombre de la columna en 'self.data' cuyos valores se van a estandarizar.
        # - 'link': La ruta al archivo de texto que contiene los mapeos de estandarización.
        #           Se espera que este archivo tenga un formato como:
        #           valor_estandar1:variacion1_1,variacion1_2,variacion1_3
        #           valor_estandar2:variacion2_1,variacion2_2
        #           ...

        # 1. Validación de existencia de la columna:
        # Comprueba si la columna especificada por 'column_name' existe en el DataFrame 'self.data'.
        if column_name not in self.data.columns:
            # Si la columna no existe, lanza un error de tipo KeyError con un mensaje informativo.
            raise KeyError(f"La columna {column_name} no existe en el DataFrame.")
        
        # 2. Carga del archivo de mapeo:
        # Inicializa un diccionario vacío para almacenar los mapeos.
        # Las claves serán los valores estándar y los valores serán listas de sus variaciones.
        mapping = {}

        try:
            # Intenta abrir y leer el archivo especificado por 'link'.
            # Se usa 'encoding="utf-8"' para asegurar la correcta lectura de caracteres especiales.
            with open(link, encoding="utf-8") as f:
                # Itera sobre cada línea del archivo.
                for line in f:
                    # Para cada línea:
                    # - line.strip(): Elimina espacios en blanco al principio y al final de la línea.
                    # - .split(":"): Divide la línea en dos partes usando el carácter ':' como delimitador.
                    #   Se espera que la primera parte sea la clave (valor estándar) y la segunda
                    #   las variaciones separadas por comas.
                    key, val = line.strip().split(":")
                    
                    # Almacena el mapeo en el diccionario:
                    # - key.lower(): La clave (valor estándar) se convierte a minúsculas.
                    # - [v.strip().lower() for v in val.split(",")]: Las variaciones (separadas por ',')
                    #   se procesan: cada variación 'v' se limpia de espacios (.strip()) y se convierte
                    #   a minúsculas (.lower()). El resultado es una lista de variaciones en minúsculas.
                    mapping[key.lower()] = [v.strip().lower() for v in val.split(",")]
        except FileNotFoundError:
            # Si el archivo especificado en 'link' no se encuentra, lanza un FileNotFoundError.
            raise FileNotFoundError(f"El archivo {link} no se encontró.")
        except Exception as e:
            # Si ocurre cualquier otro error durante la lectura o procesamiento del archivo de mapeo,
            # lanza un ValueError con el mensaje de error original.
            raise ValueError(f"Hubo un error al procesar el archivo de mapeo: {e}")

        # 3. Estandarización de los valores en la columna del DataFrame:
        # Itera sobre cada valor en la columna 'column_name' del DataFrame 'self.data'.
        # 'idvalue' será el índice de la fila y 'value' será el valor en esa fila y columna.
        # Nota: Iterar sobre una Serie y luego usar .loc[idvalue, column_name] para actualizar
        # puede ser menos eficiente en pandas que usar métodos vectorizados o .apply().
        # Sin embargo, para este tipo de lógica de reemplazo basada en un diccionario de mapeo complejo,
        # a veces es más claro implementarlo con un bucle.
        for idvalue, value in enumerate(self.data[column_name]):
            # Asegurarse de que el valor sea un string antes de procesarlo
            if not isinstance(value, str):
                continue # Saltar valores no string

            # Procesa el valor actual de la columna:
            # - value.strip(): Elimina espacios en blanco.
            # - .lower(): Convierte a minúsculas para una comparación insensible a mayúsculas/minúsculas
            #   con las variaciones del archivo de mapeo.
            current_value_processed = value.strip().lower() 
            
            # Itera sobre cada par (valor estándar, lista de variaciones) en el diccionario 'mapping'.
            for standard_form, variations_list in mapping.items():
                # Comprueba si el valor procesado de la celda actual ('current_value_processed')
                # se encuentra en la lista de variaciones ('variations_list') para el 'standard_form' actual.
                if current_value_processed in variations_list:
                    # Si se encuentra una coincidencia:
                    # Actualiza el valor en el DataFrame 'self.data' en la fila 'idvalue'
                    # y columna 'column_name' al valor estándar ('standard_form').
                    self.data.loc[idvalue, column_name] = standard_form
                    # Se rompe el bucle interno porque ya se encontró el valor estándar correspondiente y se realizó el reemplazo.
                    break

    def oversample(self, target):
        # Este método toma dos argumentos:
        # - 'self': La referencia a la instancia de la clase.
        # - 'target': Un string con el nombre de la columna en 'self.data' que representa
        #             la variable objetivo (la que se quiere predecir y está desbalanceada).

        # 1. Separar características (X) y variable objetivo (y):
        # 'y' contendrá la Serie de la columna objetivo.
        y = self.data[target]
        # 'X' contendrá un nuevo DataFrame con todas las columnas excepto la columna objetivo.
        # axis=1 indica que 'target' es una columna a eliminar.
        X = self.data.drop(target, axis=1)

        # 2. Identificar tipos de columnas en las características (X):
        # 'categorical_cols' será una lista con los nombres de las columnas en X
        # que son de tipo 'object' (generalmente strings) o 'category'.
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        # 'numeric_cols' será una lista con los nombres de las columnas en X
        # que son de tipo numérico (int, float, etc.).
        numeric_cols = X.select_dtypes(include=['number']).columns
        
        # Inicializar la variable que contendrá el estimador SMOTE.
        smote_estimator = None

        # 3. Seleccionar el estimador SMOTE apropiado basado en los tipos de características:
        
        # Caso 1: Hay tanto características numéricas como categóricas.
        if len(numeric_cols) > 0 and len(categorical_cols) > 0:
            print("Aplicando SMOTENC (para características mixtas numéricas y categóricas).")
            # 'categorical_indices' obtiene los índices posicionales de las columnas categóricas
            # dentro del DataFrame X. SMOTENC necesita estos índices.
            # Es importante que X sea un DataFrame para que .columns.get_loc funcione.
            categorical_indices = [X.columns.get_loc(col) for col in categorical_cols]
            
            smote_estimator = SMOTENC(
                categorical_features=categorical_indices,
                random_state=45, 
            )
        # Caso 2: Solo hay características numéricas (y al menos una).
        elif len(numeric_cols) > 0:
            print("Aplicando SMOTE (para características puramente numéricas).")
            smote_estimator = SMOTE(
                random_state=45
            )
        # Caso 3: Solo hay características categóricas (y al menos una), o no hay numéricas.
        # SMOTEN es para datos puramente categóricos.
        elif len(categorical_cols) > 0:
            print("Aplicando SMOTEN (para características puramente categóricas).")
            # SMOTEN espera que todas las características que se le pasen sean categóricas.
            # Si X contiene solo las columnas categóricas, esto está bien.
            # Si X pudiera tener otros tipos que no son 'object'/'category'
            # pero tampoco 'number', SMOTEN podría no ser el adecuado o necesitar preprocesamiento.
            smote_estimator = SMOTEN(
                random_state=45
            )
        else:
            # Caso 4: No hay características numéricas ni categóricas (o X está vacío).
            print("No se encontraron características numéricas ni categóricas para aplicar SMOTE/SMOTENC/SMOTEN.")
            # En este caso, no se realiza el sobremuestreo y los datos originales se mantienen.
            # No se asigna smote_estimator, por lo que el siguiente paso fallaría si no se maneja.
            print("No se realizará el sobremuestreo.")
            return # Salir del método si no hay nada que remuestrear.

        # 4. Aplicar el sobremuestreo:
        # 'fit_resample' ajusta el estimador SMOTE a X e y, y luego genera nuevas muestras
        # para la clase minoritaria para balancear el conjunto de datos.
        # X_resampled: Características sobremuestreadas.
        # y_resampled: Variable objetivo sobremuestreada.
        # Si X es un DataFrame de pandas, X_resampled será un array NumPy.
        # y_resampled también será un array NumPy.
        X_resampled, y_resampled = smote_estimator.fit_resample(X, y)

        # 5. Reconstruir el DataFrame 'self.data':
        # Se crea un nuevo DataFrame a partir de X_resampled, usando los nombres de columna originales de X.
        # Se crea una nueva Serie a partir de y_resampled, con el nombre de la columna objetivo original.
        # Luego, estos dos se concatenan a lo largo del eje de las columnas (axis=1)
        # para formar el nuevo DataFrame 'self.data' balanceado.
        self.data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=target)], axis=1)

    def encode(self, link):
        # Este método toma dos argumentos:
        # - 'self': La referencia a la instancia de la clase.
        # - 'link': Una cadena de texto (string) que es la ruta al archivo de configuración
        #           que especifica qué columnas codificar y cómo.

        # Inicializa un diccionario vacío llamado 'encoders'.
        # Este diccionario se usará para almacenar los codificadores ajustados
        # para cada columna que se procese. La clave será el nombre de la columna original
        # y el valor será la instancia del codificador ajustado.
        encoders = {}

        try:
            # Intenta abrir y leer el archivo de configuración especificado por 'link'.
            # Se usa 'encoding="utf-8"' para manejar correctamente caracteres especiales.
            with open(link, encoding="utf-8") as f:
                # Inicializa 'enc_type' a None. Esta variable almacenará el tipo de
                # codificación actual que se debe aplicar a las siguientes columnas
                # especificadas en el archivo.
                enc_type = None

                # Itera sobre cada línea del archivo de configuración.
                for line in f:
                    # Elimina espacios en blanco al principio y al final de la línea.
                    line = line.strip()

                    # Comprueba si la línea actual especifica un tipo de codificación.
                    # Los tipos de codificación válidos son "OHE" (One-Hot Encoding),
                    # "OE" (Ordinal Encoding), "LE" (Label Encoding), "CV" (Count Vectorization).
                    if line in {"OHE", "OE", "LE", "CV"}:
                        # Si la línea es un tipo de codificador, actualiza 'enc_type'.
                        # Todas las columnas que aparezcan después de esta línea se codificarán
                        # con el método seleccionado.
                        enc_type = line
                    # Comprueba si la línea especifica un nombre de columna.
                    # Se asume que los nombres de columna están precedidos por '#'.
                    elif line.startswith("#"):
                        # Extrae el nombre de la columna eliminando el '#' y los espacios.
                        col = line.replace("#", "").strip()

                        # --- Label Encoding (LE) ---
                        if enc_type == "LE":
                            # Valida que la columna exista en el DataFrame.
                            if col not in self.data.columns:
                                raise KeyError(f"La columna '{col}' no existe en el DataFrame.")

                            # Crea una instancia de LabelEncoder.
                            le = LabelEncoder()
                            # Ajusta el LabelEncoder a la columna y transforma la columna.
                            # Reemplaza la columna original con la versión codificada.
                            # LabelEncoder espera una entrada 1D.
                            self.data[col] = le.fit_transform(self.data[col])
                            # Almacena el codificador ajustado en el diccionario 'encoders'.
                            encoders[col] = le
                        
                        # --- Ordinal Encoding (OE) ---
                        elif enc_type == "OE":
                            categories = None # Inicializa las categorías como None
                            # Permite especificar un orden de categorías personalizado en el archivo.
                            # Formato esperado: #nombre_columna:categoria1;categoria2;categoria3
                            if ":" in col:
                                col, cat_str = col.split(":", 1) # Divide solo en el primer ':'
                                col = col.strip() # Limpiar nombre de columna
                                # Las categorías se pasan como una lista de listas.
                                categories = [cat_str.strip().split(";")] 

                            # Valida que la columna (después de extraer categorías opcionales) exista.
                            if col not in self.data.columns:
                                raise KeyError(f"La columna '{col}' no existe en el DataFrame.")
                            
                            # Crea una instancia de OrdinalEncoder.
                            # Si se especificaron categorías, se usan; de lo contrario, se infieren.
                            oe = OrdinalEncoder(categories=categories) if categories else OrdinalEncoder()
                            # Ajusta y transforma la columna. OrdinalEncoder espera una entrada 2D (DataFrame).
                            self.data[col] = oe.fit_transform(self.data[[col]])
                            # Almacena el codificador ajustado.
                            encoders[col] = oe
                        
                        # --- One-Hot Encoding (OHE) ---
                        elif enc_type == "OHE":
                            # Valida que la columna exista.
                            if col not in self.data.columns:
                                raise KeyError(f"La columna '{col}' no existe en el DataFrame.")
                            
                            # Crea una instancia de OneHotEncoder.
                            # sparse_output=False devuelve un array NumPy denso en lugar de una matriz dispersa.
                            # handle_unknown='ignore' hace que las categorías no vistas durante fit_transform
                            # se codifiquen como todas ceros, en lugar de lanzar un error.
                            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                            # Ajusta y transforma la columna. OHE espera una entrada 2D.
                            transformed = ohe.fit_transform(self.data[[col]])
                            # Obtiene los nombres de las nuevas columnas generadas por OHE.
                            new_ohe_columns = ohe.get_feature_names_out([col])
                            # Crea un DataFrame con los datos transformados y los nuevos nombres de columna.
                            # Se mantiene el índice original del DataFrame.
                            df_ohe = pd.DataFrame(transformed, columns=new_ohe_columns, index=self.data.index)
                            # Elimina la columna categórica original del DataFrame.
                            # Y luego une (concatena por columnas) el DataFrame original (sin la columna)
                            # con el DataFrame que contiene las nuevas columnas codificadas en one-hot.
                            self.data = self.data.drop(columns=[col]).join(df_ohe)
                            # Almacena el codificador ajustado.
                            encoders[col] = ohe # Se guarda el OHE original para posible inverse_transform
                                                # aunque la decodificación de OHE es más compleja que solo llamar a inverse_transform.
                        
                        # --- Count Vectorization (CV) ---
                        # Este método se usa típicamente para convertir texto en una matriz de conteos de tokens (palabras).
                        elif enc_type == "CV":
                            # Valida que la columna exista.
                            if col not in self.data.columns:
                                raise KeyError(f"La columna '{col}' no existe en el DataFrame.")

                            # Combina listas de stopwords en español e inglés.
                            # stopwords son palabras comunes que generalmente se eliminan antes de procesar texto.
                            stopwords_combinadas = list(stopwords.words('spanish')) + list(stopwords.words('english'))
                            stopwords_combinadas = list(set(stopwords_combinadas)) # Eliminar duplicados


                            # Crea una instancia de CountVectorizer.
                            # strip_accents='ascii' elimina acentos.
                            # stop_words=stopwords_combinadas usa la lista de stopwords definida.
                            cv = CountVectorizer(strip_accents = 'ascii', stop_words=stopwords_combinadas)
                            # Ajusta y transforma la columna de texto.
                            # .astype(str) asegura que todos los datos en la columna sean strings.
                            transformed = cv.fit_transform(self.data[col].astype(str))
                            # Crea un DataFrame con la matriz de conteo de tokens.
                            # .toarray() convierte la matriz dispersa de salida de CV a una densa.
                            df_cv = pd.DataFrame(transformed.toarray(), columns=cv.get_feature_names_out(), index=self.data.index)
                            # Elimina la columna de texto original y une el DataFrame con las nuevas columnas de conteo.
                            self.data = self.data.drop(columns=[col]).join(df_cv)
                            # Almacena el vectorizador ajustado.
                            encoders[col] = cv

        # Manejo de errores para la apertura y lectura del archivo.
        except FileNotFoundError:
            # Si el archivo no se encuentra, lanza un FileNotFoundError.
            raise FileNotFoundError(f"El archivo {link} no se encuentra.")
        except Exception as e:
            # Si ocurre cualquier otro error durante el procesamiento del archivo o la codificación,
            # lanza un ValueError con un mensaje descriptivo que incluye el error original.
            # Esto ayuda a diagnosticar problemas con el formato del archivo de configuración
            # o con los datos mismos.
            raise ValueError(f"Hubo un error al procesar el archivo de configuración o al codificar: {e}")      

        # Devuelve el diccionario 'encoders' que contiene todos los codificadores ajustados.
        # Esto permite al código que llama a este método tener acceso a los codificadores
        # para, por ejemplo, aplicar transformaciones inversas (decodificar) más tarde.
        return encoders