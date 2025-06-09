import pandas as pd
from imblearn.over_sampling import SMOTENC, SMOTE, SMOTEN

def readDataframe(link: str, encoding='utf-8', delimiter=';') -> pd.DataFrame:
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
    df = pd.read_csv(link, **read_kwargs)
    
    # Este bucle itera sobre todas las columnas del DataFrame 'data'
    # que son de tipo 'object' o 'string'. Estos tipos de datos suelen usarse
    # en pandas para almacenar texto.
    for col in df.select_dtypes(include=['object', 'string']):
        # Para cada una de estas columnas ('col'):
        # 1. .astype(str): Se asegura de que todos los valores en la columna sean tratados como strings.
        #    Esto es útil si la columna tiene tipos mixtos o valores NaN que podrían causar problemas
        #    con las operaciones de string.
        # 2. .str.strip(): Elimina los espacios en blanco al principio y al final de cada string
        #    en la columna.
        # 3. .str.lower(): Convierte todos los caracteres de cada string en la columna a minúsculas.
        df[col] = df[col].astype(str).str.strip().str.lower()

    for col in df.columns:
        # Para cada columna ('col'):
        # Intenta convertir los valores de la columna a un tipo numérico (float o int)
        # usando pd.to_numeric.
        # - 'errors='ignore'': Si un valor en la columna no puede convertirse a número,
        #   esta opción hace que pd.to_numeric no lance un error, sino que deje el valor
        #   original sin convertir. Las columnas que sí puedan convertirse completamente
        #   a numéricas cambiarán su tipo de dato (dtype); las que no, permanecerán
        #   como tipo 'object' o el tipo que tenían.
        df[col] = pd.to_numeric(df[col], errors='ignore')
    
    return df

def standarize (df: pd.DataFrame, column_name, link) -> pd.DataFrame:
    # Este método toma tres argumentos:
    # - 'df': El Dataframe de la cual se va a estandarizar una columna.
    # - 'column_name': El nombre de la columna en 'df' cuyos valores se van a estandarizar.
    # - 'link': La ruta al archivo de texto que contiene los mapeos de estandarización.
    #           Se espera que este archivo tenga un formato como:
    #           valor_estandar1:variacion1_1,variacion1_2,variacion1_3
    #           valor_estandar2:variacion2_1,variacion2_2
    #           ...

    # 1. Validación de existencia de la columna:
    # Comprueba si la columna especificada por 'column_name' existe en el DataFrame 'df'.
    if column_name not in df.columns:
        # Si la columna no existe, lanza un error de tipo KeyError con un mensaje informativo.
        raise KeyError(f"La columna {column_name} no existe en el DataFrame.")
    
    df_copy = df.copy()

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
    # Itera sobre cada valor en la columna 'column_name' del DataFrame 'df_copy'.
    # 'idvalue' será el índice de la fila y 'value' será el valor en esa fila y columna.
    # Nota: Iterar sobre una Serie y luego usar .loc[idvalue, column_name] para actualizar
    # puede ser menos eficiente en pandas que usar métodos vectorizados o .apply().
    # Sin embargo, para este tipo de lógica de reemplazo basada en un diccionario de mapeo complejo,
    # a veces es más claro implementarlo con un bucle.
    for idvalue, value in enumerate(df_copy[column_name]):
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
                # Actualiza el valor en el DataFrame 'df_copy' en la fila 'idvalue'
                # y columna 'column_name' al valor estándar ('standard_form').
                df_copy.loc[idvalue, column_name] = standard_form
                # Se rompe el bucle interno porque ya se encontró el valor estándar correspondiente y se realizó el reemplazo.
                break
    
    return df_copy

def oversample(df: pd.DataFrame, target: str) -> pd.DataFrame:
    # Este método toma dos argumentos:
    # - 'df': El marco de datos que se va a sobremuestrear.
    # - 'target': Un string con el nombre de la columna en 'df' que representa
    #             la variable objetivo (la que se quiere predecir y está desbalanceada).

    # 1. Separar características (X) y variable objetivo (y):
    # 'y' contendrá la Serie de la columna objetivo.
    y = df[target]
    # 'X' contendrá un nuevo DataFrame con todas las columnas excepto la columna objetivo.
    # axis=1 indica que 'target' es una columna a eliminar.
    X = df.drop(target, axis=1)

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

    # 5. Reconstruir el DataFrame:
    # Se crea un nuevo DataFrame a partir de X_resampled, usando los nombres de columna originales de X.
    # Se crea una nueva Serie a partir de y_resampled, con el nombre de la columna objetivo original.
    # Luego, estos dos se concatenan a lo largo del eje de las columnas (axis=1).
    return pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=target)], axis=1)
