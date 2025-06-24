import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from typing import Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
import nltk
try:
    from nltk.corpus import stopwords
    stopwords.words('spanish')  # o 'english', según idioma
except LookupError:
    nltk.download('stopwords')

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def encode(df: pd.DataFrame, link: str) -> Tuple[pd.DataFrame, dict]:
        # Este método toma dos argumentos:
        # - 'df': El marco de datos que se va a codificar.
        # - 'link': Una cadena de texto (string) que es la ruta al archivo de configuración
        #           que especifica qué columnas codificar y cómo.

        # Inicializa un diccionario vacío llamado 'encoders'.
        # Este diccionario se usará para almacenar los codificadores ajustados
        # para cada columna que se procese. La clave será el nombre de la columna original
        # y el valor será la instancia del codificador ajustado.
        df_copy = df.copy()
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
                            if col not in df_copy.columns:
                                continue
                                #raise KeyError(f"La columna '{col}' no existe en el DataFrame.")

                            # Crea una instancia de LabelEncoder.
                            le = LabelEncoder()
                            # Ajusta el LabelEncoder a la columna y transforma la columna.
                            # Reemplaza la columna original con la versión codificada.
                            # LabelEncoder espera una entrada 1D.
                            df_copy[col] = le.fit_transform(df_copy[col])
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
                            if col not in df_copy.columns:
                                continue
                                #raise KeyError(f"La columna '{col}' no existe en el DataFrame.")
                            
                            # Crea una instancia de OrdinalEncoder.
                            # Si se especificaron categorías, se usan; de lo contrario, se infieren.
                            oe = OrdinalEncoder(categories=categories) if categories else OrdinalEncoder()
                            # Ajusta y transforma la columna. OrdinalEncoder espera una entrada 2D (DataFrame).
                            df_copy[col] = oe.fit_transform(df_copy[[col]])
                            # Almacena el codificador ajustado.
                            encoders[col] = oe
                        
                        # --- One-Hot Encoding (OHE) ---
                        elif enc_type == "OHE":
                            # Valida que la columna exista.
                            if col not in df_copy.columns:
                                continue
                                #raise KeyError(f"La columna '{col}' no existe en el DataFrame.")
                            
                            # Crea una instancia de OneHotEncoder.
                            # sparse_output=False devuelve un array NumPy denso en lugar de una matriz dispersa.
                            # handle_unknown='ignore' hace que las categorías no vistas durante fit_transform
                            # se codifiquen como todas ceros, en lugar de lanzar un error.
                            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                            # Ajusta y transforma la columna. OHE espera una entrada 2D.
                            transformed = ohe.fit_transform(df_copy[[col]])
                            # Obtiene los nombres de las nuevas columnas generadas por OHE.
                            new_ohe_columns = ohe.get_feature_names_out([col])
                            # Crea un DataFrame con los datos transformados y los nuevos nombres de columna.
                            # Se mantiene el índice original del DataFrame.
                            df_ohe = pd.DataFrame(transformed, columns=new_ohe_columns, index=df_copy.index)
                            # Elimina la columna categórica original del DataFrame.
                            # Y luego une (concatena por columnas) el DataFrame original (sin la columna)
                            # con el DataFrame que contiene las nuevas columnas codificadas en one-hot.
                            df_copy = df_copy.drop(columns=[col]).join(df_ohe)
                            # Almacena el codificador ajustado.
                            encoders[col] = ohe # Se guarda el OHE original para posible inverse_transform
                                                # aunque la decodificación de OHE es más compleja que solo llamar a inverse_transform.
                        
                        # --- Count Vectorization (CV) ---
                        # Este método se usa típicamente para convertir texto en una matriz de conteos de tokens (palabras).
                        elif enc_type == "CV":
                            # Valida que la columna exista.
                            if col not in df_copy.columns:
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
                            transformed = cv.fit_transform(df_copy[col].astype(str))
                            # Crea un DataFrame con la matriz de conteo de tokens.
                            # .toarray() convierte la matriz dispersa de salida de CV a una densa.
                            df_cv = pd.DataFrame(transformed.toarray(), columns=cv.get_feature_names_out(), index=df_copy.index)
                            # Elimina la columna de texto original y une el DataFrame con las nuevas columnas de conteo.
                            df_copy = df_copy.drop(columns=[col]).join(df_cv)
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
        return df_copy, encoders

def decode(df: pd.DataFrame, encoders: dict, target: str) -> pd.DataFrame:
    # Su objetivo es decodificar las columnas del DataFrame 'df'
    # utilizando los codificadores proporcionados. Argumentos:
    # - 'df': Un DataFrame de pandas que contiene los datos codificados que se van a decodificar.
    # - 'encoders': Un diccionario donde las claves son los nombres de las columnas originales
    #               y los valores son las instancias de los codificadores ajustados que se usaron para codificar.
    # - 'target': La columna objetivo a la que se le cambiará la posición al final para su posterior muetreo.

    df_copy = df.copy()

    # Itera sobre cada par en el diccionario 'encoders'.
    for col, encoder in encoders.items():
        # Comprueba el tipo de codificador para aplicar la decodificación correcta.

        # --- LabelEncoder ---
        if isinstance(encoder, LabelEncoder):
            # Si la columna 'col' existe en el DataFrame 'df_copy'.
            if col in df_copy.columns:
                # LabelEncoder.inverse_transform espera valores numéricos (enteros).
                # Se asegura de que la columna sea de tipo int64 antes de decodificar.
                # Esto podría ser problemático si los valores no son realmente enteros
                # o si LIME/DiCE los ha perturbado a flotantes.
                df_copy[col] = df_copy[col].astype('int64') 
                # Aplica la transformación inversa para obtener las etiquetas originales.
                df_copy[col] = encoder.inverse_transform(df_copy[col])
            else:
                print(f"Advertencia: Columna '{col}' no encontrada en los datos para decodificar con LabelEncoder.")

        # --- OrdinalEncoder ---
        elif isinstance(encoder, OrdinalEncoder):
            # Si la columna 'col' existe en el DataFrame 'df_copy'.
            if col in df_copy.columns:
                # OrdinalEncoder.inverse_transform espera una entrada 2D (como un DataFrame de una columna).
                # El resultado también es 2D, por lo que .ravel() lo convierte a 1D para asignarlo
                # de nuevo a la columna del DataFrame.
                df_copy[col] = encoder.inverse_transform(df_copy[[col]]).ravel()
            else:
                print(f"Advertencia: Columna '{col}' no encontrada en los datos para decodificar con OrdinalEncoder.")

        # --- OneHotEncoder ---
        elif isinstance(encoder, OneHotEncoder):
            # 'col' aquí es el nombre de la columna *original* antes de que se expandiera con OHE.
            # Se obtienen los nombres de las columnas que fueron generadas por el OneHotEncoder
            # para esta característica original 'col'.
            ohe_cols = encoder.get_feature_names_out([col])
            
            # Comprobar si todas las columnas OHE necesarias existen en el DataFrame 'df_copy'.
            if all(c in df_copy.columns for c in ohe_cols):
                # OneHotEncoder.inverse_transform espera una entrada 2D que contenga
                # las columnas codificadas en one-hot.
                # El resultado es un array 2D de forma (n_muestras, 1) con las categorías originales.
                # Se usa .ravel() o [:, 0] para obtener un array 1D.
                original_categories = encoder.inverse_transform(df_copy[ohe_cols]).ravel()
                
                # Se elimina las columnas one-hot originales.
                df_copy = df_copy.drop(columns=ohe_cols)
                # Se crea una nueva columna con el nombre original 'col' y se le asignan
                # las categorías decodificadas.
                # Es importante asegurarse de que el índice se alinee correctamente.
                df_copy[col] = pd.Series(original_categories, index=df_copy.index) 
            else:
                missing_ohe_cols = [c for c in ohe_cols if c not in df_copy.columns]
                print(f"Advertencia: Faltan columnas OHE ({missing_ohe_cols}) para la característica original '{col}' en los datos. No se puede decodificar.")

        # --- CountVectorizer ---
        elif isinstance(encoder, CountVectorizer):
            # 'col' es el nombre de la columna de texto original que fue vectorizada.
            # 'df_copy' en este punto debería contener las columnas generadas por CountVectorizer (la matriz término-documento).
            # Sin embargo, el bucle itera sobre 'encoders' donde la clave 'col' es el nombre de la columna *original*.
            # Esto implica que 'df_copy' debería ser la matriz de características vectorizadas, y 'col'
            # el nombre que se le quiere dar a la columna de texto "reconstruida".

            # inverse_transform(X) toma una matriz término-documento X
            # y devuelve una lista de arrays, donde cada array contiene los tokens
            # presentes en el documento correspondiente.
            
            # El DataFrame 'df_copy' que se pasa a este método debe ser la matriz de conteo
            # si queremos aplicar inverse_transform. Las siguientes líneas intentan una
            # reconstrucción que probablemente no sea la deseada para revertir CountVectorizer a una columna de texto original.
            
            # Obtener los nombres de todas las características que generó el CountVectorizer
            cv_feature_names = encoder.get_feature_names_out()

            # Comprobar si todas las columnas del CV existen en el DataFrame 'df_copy'
            if all(c in df_copy.columns for c in cv_feature_names):
                # 'transformed' será una lista de arrays de strings (tokens)
                transformed_tokens_list = encoder.inverse_transform(df_copy[cv_feature_names])
                
                # Unir los tokens de cada documento para formar una cadena de texto
                # Esto es una forma de "reconstruir" el texto, aunque no será idéntico al original
                reconstructed_text = [" ".join(tokens) for tokens in transformed_tokens_list]

                # Eliminar todas las columnas generadas por CountVectorizer.
                df_copy = df_copy.drop(columns=cv_feature_names)
                # Añadir la columna 'col' (nombre original de la característica de texto) con el texto reconstruido.
                df_copy[col] = pd.Series(reconstructed_text, index=df_copy.index)
            else:
                missing_cv_cols = [c for c in cv_feature_names if c not in df_copy.columns]
                print(f"Advertencia: Faltan columnas de CountVectorizer ({missing_cv_cols}) para la característica original '{col}'. No se puede decodificar.")

    # --- Mover la columna objetivo al final ---
    # Obtiene una lista de todos los nombres de columna actuales en el DataFrame 'df_copy'.
    cols = list(df_copy.columns)
    # Comprueba si la columna objetivo existe en el DataFrame.
    if target in cols:
        # Si existe, mueve la columna objetivo al final de la lista de columnas.
        # 1. Encuentra el índice de la columna objetivo.
        # 2. Intercambia la columna objetivo con la última columna en la lista 'cols'.
        #    Esto se hace mediante asignación múltiple.
        target_idx = cols.index(target)
        cols[target_idx], cols[-1] = cols[-1], cols[target_idx]
        
        # Reordena el DataFrame 'df_copy' según el nuevo orden de columnas en 'cols'.
        df_copy = df_copy[cols]

    # Devuelve el DataFrame 'df_copy' con las columnas decodificadas y reordenadas.
    return df_copy