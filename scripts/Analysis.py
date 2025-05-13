import pandas as pd
import dice_ml
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
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
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

    def __init__(self, link, encoding, delimiter):
        read_kwargs = {'encoding': encoding, 'delimiter': delimiter}
        read_kwargs = {k: v for k, v in read_kwargs.items() if v is not None}
        self.data = pd.read_csv(link, **read_kwargs)
        
        for col in self.data.select_dtypes(include=['object', 'string']):
            self.data[col] = self.data[col].astype(str).str.strip().str.lower()

        for col in self.data.columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='ignore')

    def add(self, column_name, values):
        if isinstance(values, str) or not hasattr(values, '__len__'):
            raise TypeError("El argumento 'values' debe ser una lista, Serie u otro iterable, no un string.")

        if len(values) == len(self.data):
            self.data[column_name] = values
        else:
            raise ValueError("La longitud del array no coincide con el número de filas del DataFrame.")
    
    def delete(self, columns):
        if isinstance(columns, str):
            columns = [columns]

        missing = [col for col in columns if col not in self.data.columns]
        if missing:
            raise KeyError(f"Las siguientes columnas no se encontraron en el DataFrame: {missing}")

        self.data.drop(columns=columns, inplace=True)

    def replace(self, column1, column1_value, column2, column2_value, new_column2_value, case_sensitive=False):
        if column1 not in self.data.columns or column2 not in self.data.columns:
            raise KeyError(f"Las columnas {column1} o {column2} no existen en el DataFrame.")

        if not case_sensitive:
            column1_value = column1_value.lower()
            column2_value = column2_value.lower()

        self.data.loc[
            (self.data[column1].apply(lambda x: x.lower() if isinstance(x, str) else x) == column1_value) &
            (self.data[column2].apply(lambda x: x.lower() if isinstance(x, str) else x) == column2_value), 
            column2
        ] = new_column2_value
    
    def standarize (self, column_name, link):
        if column_name not in self.data.columns:
            raise KeyError(f"La columna {column_name} no existe en el DataFrame.")
        
        mapping = {}

        try:
            with open(link, encoding="utf-8") as f:
                for line in f:
                    key, val = line.strip().split(":")
                    mapping[key.lower()] = [v.strip().lower() for v in val.split(",")]
        except FileNotFoundError:
            raise FileNotFoundError(f"El archivo {link} no se encontró.")
        except Exception as e:
            raise ValueError(f"Hubo un error al procesar el archivo: {e}")

        for idvalue, value in enumerate(self.data[column_name]):
            value = value.strip().lower() 
            for standard, variations in mapping.items():
                if value in variations:
                    self.data.loc[idvalue, column_name] = standard
                    break 

    def resample(self, target):
        y = self.data[target]
        X = self.data.drop(target, axis=1)

        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numeric_cols = X.select_dtypes(include=['number']).columns
        smote_estimator = None

        if len(numeric_cols) > 0 and len(categorical_cols) > 0:
            categorical_indices = [X.columns.get_loc(col) for col in categorical_cols]
            smote_estimator = SMOTENC(
                categorical_features=categorical_indices,
                random_state=45
            )
        elif len(numeric_cols) > 0:
            smote_estimator = SMOTE(random_state=45)
        else:
            smote_estimator = SMOTEN(random_state=45)

        X_resampled, y_resampled = smote_estimator.fit_resample(X, y)
        self.data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=target)], axis=1)

    def encode(self, link):
        encoders = {}

        try:
            with open(link, encoding="utf-8") as f:
                enc_type = None

                for line in f:
                    line = line.strip()

                    if line in {"OHE", "OE", "LE", "CV"}:
                        enc_type = line
                    elif line.startswith("#"):
                        col = line.replace("#", "").strip()

                        if enc_type == "LE":
                            if col not in self.data.columns:
                                raise KeyError(f"La columna '{col}' no existe en el DataFrame.")

                            le = LabelEncoder()
                            self.data[col] = le.fit_transform(self.data[col])
                            encoders[col] = le
                        elif enc_type == "OE":
                            categories = None
                            if ":" in col:
                                col, cat_str = col.split(":")
                                categories = [cat_str.split(";")]

                            if col not in self.data.columns:
                                raise KeyError(f"La columna '{col}' no existe en el DataFrame.")
                            
                            oe = OrdinalEncoder(categories=categories) if categories else OrdinalEncoder()
                            self.data[col] = oe.fit_transform(self.data[[col]])
                            encoders[col] = oe
                        elif enc_type == "OHE":

                            if col not in self.data.columns:
                                raise KeyError(f"La columna '{col}' no existe en el DataFrame.")
                            
                            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                            transformed = ohe.fit_transform(self.data[[col]])
                            columns = ohe.get_feature_names_out([col])
                            df_ohe = pd.DataFrame(transformed, columns=columns, index=self.data.index)
                            self.data = self.data.drop(columns=[col]).join(df_ohe)
                            encoders[col] = ohe
                        elif enc_type == "CV":
                            
                            if col not in self.data.columns:
                                raise KeyError(f"La columna '{col}' no existe en el DataFrame.")

                            stopwords_combinadas = list(stopwords.words('spanish')) + list(stopwords.words('english'))
                            stopwords_combinadas = list(set(stopwords_combinadas))
                            cv = CountVectorizer(stop_words=stopwords_combinadas)
                            transformed = cv.fit_transform(self.data[col].astype(str))
                            df_cv = pd.DataFrame(transformed.toarray(), columns=cv.get_feature_names_out(), index=self.data.index)
                            self.data = self.data.drop(columns=[col]).join(df_cv)
                            encoders[col] = cv

        except FileNotFoundError:
            raise FileNotFoundError(f"El archivo {link} no se encuentra.")
        except Exception as e:
            raise ValueError(f"Hubo un error al procesar el archivo: {e}")      

        return encoders

class Classification:

    def __init__(self, data, target, encoders, test_size=0.2):
        Y = data[target]
        X = data.drop(target, axis=1)

        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(X, Y, test_size=test_size, random_state=42)

        self.encoders = encoders
        self.list_models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'SVC': SVC(probability=True, random_state=42),
            'Naive Bayes': GaussianNB(),
            'MLP Classifier': MLPClassifier(max_iter=1000, random_state=42),
            'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
        }

    def trainEvaluate(self):
        records = []

        for name, model in self.list_models.items():
            try:
                model.fit(self.xtrain, self.ytrain)
                ypred = model.predict(self.xtest)
                
                accuracy = accuracy_score(self.ytest, ypred)
                precision = precision_score(self.ytest, ypred, average='weighted', zero_division=1)
                recall = recall_score(self.ytest, ypred, average='weighted')
                f1 = f1_score(self.ytest, ypred, average='weighted')
                
                record = {
                    'Model': name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1
                }
                records.append(record)
            
            except Exception as e:
                print(f"Error al entrenar o predecir con el modelo {name}: {e}")
                record = {
                    'Model': name,
                    'Accuracy': None,
                    'Precision': None,
                    'Recall': None,
                    'F1-Score': None
                }
                records.append(record)
        
        return pd.DataFrame(records).sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
    
    def train(self, models_selected):
        
        if isinstance(models_selected, str):
            models_selected = [models_selected]
        
        estimators = []

        for name in models_selected:
            if name in self.list_models:
                model = self.list_models[name]
                if name == 'SVC' and not hasattr(model, 'probability'):
                    model.probability = True
                estimators.append((name, model))
            else:
                print(f"Advertencia: El modelo '{name}' no se encuentra en la lista de modelos disponibles.")

        if not estimators:
            raise ValueError("No se ha seleccionado ningún modelo válido para el VotingClassifier.")

        self.model = VotingClassifier(estimators=estimators, voting='soft')
        self.model.fit(self.xtrain, self.ytrain)
        
        ypred = self.model.predict(self.xtest)
        
        record = {
            'Model': "VotingClassifier",
            'Accuracy': accuracy_score(self.ytest, ypred),
            'Precision': precision_score(self.ytest, ypred, average='weighted', zero_division=1),
            'Recall': recall_score(self.ytest, ypred, average='weighted'),
            'F1-Score': f1_score(self.ytest, ypred, average='weighted')
        }

        return pd.DataFrame([record])

    def predict(self, row):

        if not hasattr(self, 'model') and getattr(self, 'model') is None:
            raise ValueError("Se debe entrenar antes un modelo con el metodo train().")

        if isinstance(row, pd.Series):
            row = row.to_frame().T

        ypred = self.model.predict(row)
        yprob = self.model.predict_proba(row)
        
        row_pred = pd.concat([row, pd.DataFrame(ypred, columns=[self.ytrain.name])], axis=1)

        prob_values = [yprob[i][ypred[i]] for i in range(len(ypred))]
        prob_values = [p * 100 for p in prob_values]

        return row_pred, pd.Series(prob_values).round(2)

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
class Counterfactual:

    def __init__(self, model, data, target, continuous_features=None, method="kdtree"):
        if not hasattr(model, 'predict') or not callable(getattr(model, 'predict')):
            raise ValueError("El modelo debe tener un método 'predict'.")
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Los datos deben ser un DataFrame de pandas.")
        
        if target not in data.columns:
            raise ValueError(f"La columna objetivo '{target}' no se encuentra en los datos.")
        
        continuous_features = continuous_features or []
        
        try:
            dmodel = dice_ml.Model(model=model, backend="sklearn")
            d = dice_ml.Data(dataframe=data, continuous_features=continuous_features, outcome_name=target)
            self.exp = dice_ml.Dice(d, dmodel, method=method)
        except Exception as e:
            raise ValueError(f"Hubo un error al inicializar DiceML: {e}")
        
        self.model = model
        self.data = data
        self.target = target

    def counterfac(self, row, encoders, target_value = None):
        if not isinstance(row, pd.DataFrame):
            raise ValueError("La fila debe ser un DataFrame.")

        if target_value == None:
            target_value = "opposite"

        cf_row = row.drop(columns=[self.target])
        
        try:
            dice_exp = self.exp.generate_counterfactuals(cf_row, total_CFs=4, desired_class=target_value)
        except Exception as e:
            raise ValueError(f"Error al generar contra-factuales: {e}")
        
        cf_data = dice_exp.cf_examples_list[0].final_cfs_df
        
        cf_data = self.__decode(cf_data, encoders)
        cf_row = self.__decode(row, encoders)

        unchanged_columns = cf_data.columns[cf_data.eq(cf_row.iloc[0]).all()]
        cf_data.drop(columns=unchanged_columns, inplace=True)
        cf_row.drop(columns=unchanged_columns, inplace=True)

        print("Fila original:", cf_row)
        print("Contra-factuales generados:", cf_data)
        
        return cf_row, cf_data

    def __decode(self, data, encoders):
        for col, encoder in encoders.items():
                if isinstance(encoder, LabelEncoder):
                    data[col] = data[col].astype('int64')
                    data[col] = encoder.inverse_transform(data[col])

                elif isinstance(encoder, OrdinalEncoder):
                    data[col] = encoder.inverse_transform(data[[col]]).ravel()

                elif isinstance(encoder, OneHotEncoder):
                    ohe_cols = encoder.get_feature_names_out([col])
                    data[col] = encoder.inverse_transform(data[ohe_cols])
                    data = data.drop(columns=ohe_cols)

                elif isinstance(encoder, CountVectorizer):
                    transformed = encoder.inverse_transform(data)
                    df_cv = pd.DataFrame(transformed.toarray(), columns=encoder.get_feature_names_out(), index=data.index)
                    data = data.drop(columns=[col]).join(df_cv)

        cols = list(data.columns)
        if self.target in cols:
            cols[cols.index(self.target)], cols[-1] = cols[-1], cols[cols.index(self.target)]
            data = data[cols]

        return data
