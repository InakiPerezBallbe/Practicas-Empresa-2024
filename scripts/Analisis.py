import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import dice_ml

class Preprocesamiento:
    
    def __init__(self, link):
        self.link = link
        self.data = pd.read_csv(link)
        for col in self.data:
            self.data.loc[:, col] = self.data[col].map(lambda x: x.strip().lower() if isinstance(x, str) else x)
           
    def add(self, column_name, values):
        if len(values) == len(self.data):
            # Si tiene el mismo tamaño, se añade la columna
            self.data[column_name] = values
        else:
            raise ValueError("La longitud del array no coincide con el número de filas del DataFrame")
        
    def delete(self, columns):
        for column in columns:
            self.data.drop(columns=[column], inplace=True)

    def replace(self, column1, column1_value, column2, column2_value, new_column2_value):
        self.data.loc[(self.data[column1] == column1_value.lower()) & (self.data[column2] == column2_value.lower()), column2] = new_column2_value.lower()
    
    def standarize (self, column_name, link):
        mapping = {}

        with open(link, encoding="utf-8") as f:
            for line in f:
                (key, val) = line.split(":")
                mapping.update([(key.lower(), val.strip().lower().split(","))])

        for idvalue, value in enumerate(self.data[column_name]):
            value = value.strip().lower()
            for standard, variations in mapping.items():
                if value in map(str.lower, variations):
                    self.data.loc[idvalue, column_name] = standard.lower()
        
        self.data[column_name] = self.data[column_name].str.lower()

    def encode(self, link):
        with open(link, encoding = "utf-8") as f:

            for line in f:
                line = line.strip()

                if line == "OHE":
                    enc_type = "OHE"
                elif line == "OE":
                    enc_type = "OE"
                elif line == "LE":
                    enc_type = "LE"
                elif line.startswith("#"):
                    col = line.replace("#", "").strip()

                    if enc_type == "LE":
                        le = LabelEncoder()
                        self.data[col] = le.fit_transform(self.data[col])
                    elif enc_type == "OE":
                        categories = None
                        if ":" in col:
                            (col, categories) = col.split(":")
                            categories = [categories.split(";")]
                        oe = OrdinalEncoder(categories=categories) if categories else OrdinalEncoder()
                        self.data[col] = oe.fit_transform(self.data[[col]])
                    elif enc_type == "OHE":
                        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                        transformed = ohe.fit_transform(self.data[[col]])
                        columns = ohe.get_feature_names_out([col])
                        df_ohe = pd.DataFrame(transformed, columns=columns, index=self.data.index)
                        self.data = self.data.drop(columns=[col]).join(df_ohe)

class Modelaje:

    def __init__ (self, data, target, test_size):
        smote = SMOTE(random_state=42)
        self.X, self.Y = smote.fit_resample(data.drop(target, axis=1), data[target])
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.X, self.Y, test_size=test_size, random_state=42)

    def metrics(self):
        print("Metricas de "+type(self.model).__name__+"\n"+
            "\tAccuracy: "+str(self.accuracy)+"\n"+
            "\tPrecision: "+str(self.precision)+"\n"+
            "\tRecall: "+str(self.recall)+"\n"+
            "\tF1-score: "+str(self.f1)+"\n")
    
    def logisticRegresion(self):
        self.model = LogisticRegression(random_state=42).fit(self.X, self.Y)

        (attributes, attribute_importance) = (self.model.feature_names_in_, pd.Series(abs(self.model.coef_[0]), index=self.X.columns))
        attributes = list(attributes[attribute_importance < 0.01])

        xtrain = pd.DataFrame(self.xtrain).drop(attributes, axis=1)
        xtest = pd.DataFrame(self.xtest).drop(attributes, axis=1)

        self.model.fit(xtrain, self.ytrain)
        ypred = self.model.predict(xtest)

        self.accuracy = accuracy_score(self.ytest, ypred)
        self.precision = precision_score(self.ytest, ypred)
        self.recall = recall_score(self.ytest, ypred)
        self.f1 = f1_score(self.ytest, ypred)

    def decisionTree(self):
        self.model = DecisionTreeClassifier(random_state=42).fit(self.X, self.Y)

        (attributes, attribute_importance) = (self.model.feature_names_in_, pd.Series(self.model.feature_importances_, index=self.X.columns))
        attributes = list(attributes[attribute_importance < 0.01])

        xtrain = pd.DataFrame(self.xtrain).drop(attributes, axis=1)
        xtest = pd.DataFrame(self.xtest).drop(attributes, axis=1)

        self.model.fit(xtrain, self.ytrain)
        ypred = self.model.predict(xtest)

        self.accuracy = accuracy_score(self.ytest, ypred)
        self.precision = precision_score(self.ytest, ypred)
        self.recall = recall_score(self.ytest, ypred)
        self.f1 = f1_score(self.ytest, ypred)

    def randomForest(self):
        self.model = RandomForestClassifier(random_state=42).fit(self.X, self.Y)

        (attributes, attribute_importance) = (self.model.feature_names_in_, pd.Series(self.model.feature_importances_, index=self.X.columns))
        attributes = list(attributes[attribute_importance < 0.01])
        
        xtrain = pd.DataFrame(self.xtrain).drop(attributes, axis=1)
        xtest = pd.DataFrame(self.xtest).drop(attributes, axis=1)

        self.model.fit(xtrain, self.ytrain)
        ypred = self.model.predict(xtest)

        self.accuracy = accuracy_score(self.ytest, ypred)
        self.precision = precision_score(self.ytest, ypred)
        self.recall = recall_score(self.ytest, ypred)
        self.f1 = f1_score(self.ytest, ypred)

    def kNeighbors(self):
        self.model = KNeighborsClassifier().fit(self.xtrain, self.ytrain)
        ypred = self.model.predict(self.xtest)

        self.accuracy = accuracy_score(self.ytest, ypred)
        self.precision = precision_score(self.ytest, ypred)
        self.recall = recall_score(self.ytest, ypred)
        self.f1 = f1_score(self.ytest, ypred)

    def xgb(self):
        self.model = XGBClassifier().fit(self.X, self.Y)

        (attributes, attribute_importance) = (self.model.feature_names_in_, pd.Series(self.model.feature_importances_, index=self.X.columns))
        attributes = list(attributes[attribute_importance < 0.01])
        
        xtrain = pd.DataFrame(self.xtrain).drop(attributes, axis=1)
        xtest = pd.DataFrame(self.xtest).drop(attributes, axis=1)

        self.model.fit(xtrain, self.ytrain)
        ypred = self.model.predict(xtest)

        self.accuracy = accuracy_score(self.ytest, ypred)
        self.precision = precision_score(self.ytest, ypred)
        self.recall = recall_score(self.ytest, ypred)
        self.f1 = f1_score(self.ytest, ypred)

    def gradientBoosting(self):
        self.model = GradientBoostingClassifier().fit(self.X, self.Y)

        (attributes, attribute_importance) = (self.model.feature_names_in_, pd.Series(self.model.feature_importances_, index=self.X.columns))
        attributes = list(attributes[attribute_importance < 0.01])
        
        xtrain = pd.DataFrame(self.xtrain).drop(attributes, axis=1)
        xtest = pd.DataFrame(self.xtest).drop(attributes, axis=1)

        self.model.fit(xtrain, self.ytrain)
        ypred = self.model.predict(xtest)

        self.accuracy = accuracy_score(self.ytest, ypred)
        self.precision = precision_score(self.ytest, ypred)
        self.recall = recall_score(self.ytest, ypred)
        self.f1 = f1_score(self.ytest, ypred)

    def gaussianNB(self):
        self.model = GaussianNB().fit(self.X, self.Y)
        ypred = self.model.predict(self.xtest)

        self.accuracy = accuracy_score(self.ytest, ypred)
        self.precision = precision_score(self.ytest, ypred)
        self.recall = recall_score(self.ytest, ypred)
        self.f1 = f1_score(self.ytest, ypred)

    def svc(self):
        self.model = SVC(random_state=42).fit(self.X, self.Y)
        ypred = self.model.predict(self.xtest)

        self.accuracy = accuracy_score(self.ytest, ypred)
        self.precision = precision_score(self.ytest, ypred)
        self.recall = recall_score(self.ytest, ypred)
        self.f1 = f1_score(self.ytest, ypred)

    def adaBoost(self):
        self.model = AdaBoostClassifier(random_state=42).fit(self.X, self.Y)

        (attributes, attribute_importance) = (self.model.feature_names_in_, pd.Series(self.model.feature_importances_, index=self.X.columns))
        attributes = list(attributes[attribute_importance < 0.01])
        
        xtrain = pd.DataFrame(self.xtrain).drop(attributes, axis=1)
        xtest = pd.DataFrame(self.xtest).drop(attributes, axis=1)

        self.model.fit(xtrain, self.ytrain)
        ypred = self.model.predict(xtest)

        self.accuracy = accuracy_score(self.ytest, ypred)
        self.precision = precision_score(self.ytest, ypred)
        self.recall = recall_score(self.ytest, ypred)
        self.f1 = f1_score(self.ytest, ypred)

    def catBoost(self):
        self.model = CatBoostClassifier(random_state=42, verbose=0).fit(self.X, self.Y)

        attribute_importance = pd.Series(self.model.feature_importances_, index=self.X.columns)
        attributes = attribute_importance[attribute_importance < 0.01].index.tolist()

        xtrain = pd.DataFrame(self.xtrain).drop(attributes, axis=1)
        xtest = pd.DataFrame(self.xtest).drop(attributes, axis=1)

        self.model.fit(xtrain, self.ytrain)
        ypred = self.model.predict(xtest)

        self.accuracy = accuracy_score(self.ytest, ypred)
        self.precision = precision_score(self.ytest, ypred)
        self.recall = recall_score(self.ytest, ypred)
        self.f1 = f1_score(self.ytest, ypred)

    def multilayerPerceptron(self):
        selector = SelectKBest(mutual_info_classif, k='all')
        selector.fit(self.X, self.Y)
        importances = selector.scores_

        attributes = pd.Series(importances, index=self.X.columns)
        attributes_to_drop = list(attributes[attributes < 0.01].index)

        attributes_to_drop = [col for col in attributes_to_drop if col in self.xtrain.columns]
        self.xtrain = self.xtrain.drop(columns=attributes_to_drop)
        self.xtest = self.xtest.drop(columns=attributes_to_drop)

        self.model = MLPClassifier(random_state=42)
        self.model.fit(self.xtrain, self.ytrain)
        ypred = self.model.predict(self.xtest)

        self.accuracy = accuracy_score(self.ytest, ypred)
        self.precision = precision_score(self.ytest, ypred)
        self.recall = recall_score(self.ytest, ypred)
        self.f1 = f1_score(self.ytest, ypred)

class Clasificacion:

    def __init__(self, model):
        self.dmodel = dice_ml.Model(model=model.model, backend="sklearn")
        d = dice_ml.Data(dataframe=pd.concat([pd.DataFrame(model.xtrain), pd.DataFrame(model.ytrain)], axis=1), continuous_features=[], outcome_name="Conciencia_Ambiental")
        exp = dice_ml.Dice(d, self.dmodel, method="random")
        query_instance = model.sxtest[0:1]
        dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=4, desired_class="opposite")

        dice_exp.visualize_as_dataframe()

    """
    # Seleccionamos una fila del conjunto de prueba
    query_instance = x_test_encoded.iloc[1:2]

    # Generamos los ejemplos contrafactuales
    dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=4, desired_class="opposite")

    # Visualizamos el contrafactual como un DataFrame
    dice_exp.visualize_as_dataframe()

    # Paso 1: Creamos un DataFrame con los valores originales
    cf_array = dice_exp.cf_examples_list[0].final_cfs_df.values
    cf_df = pd.DataFrame(cf_array)

    # Paso 2: Calculamos límites de columnas
    onehot_cols_count = ct.named_transformers_['one_hot'].get_feature_names_out().shape[0]
    ordinal_cols_count = len(categorical_cols_ordinal)
    ordinal_start = onehot_cols_count
    ordinal_end = ordinal_start + ordinal_cols_count

    # Paso 3: Descodificamos las columnas ordinales
    ordinal_encoder = ct.named_transformers_['ordinal']
    ordinal_data = cf_df.iloc[:, ordinal_start:ordinal_end]
    decoded_ordinal_values = ordinal_encoder.inverse_transform(ordinal_data)

    # Paso 4: Creamos un nuevo DataFrame combinando:
    # - las columnas antes de las ordinales
    # - las columnas ordinales decodificadas
    # - las columnas después de las ordinales
    before = cf_df.iloc[:, :ordinal_start]
    decoded = pd.DataFrame(decoded_ordinal_values, index=cf_df.index)
    after = cf_df.iloc[:, ordinal_end:]

    # Paso 5: Unimos todo en un nuevo DataFrame final
    cf_df_clean = pd.concat([before, decoded, after], axis=1)

    # Paso 6: Generar nombres de columnas
    # Para las columnas de OneHot:
    onehot_column_names = ct.named_transformers_['one_hot'].get_feature_names_out()

    # Para las columnas ordinales:
    ordinal_column_names = categorical_cols_ordinal

    # Para las columnas restantes (después de las ordinales):
    remaining_column_names = cf_df.columns[ordinal_end:].tolist()

    # Unir todos los nombres de columna
    all_column_names = list(onehot_column_names) + list(ordinal_column_names) + remaining_column_names

    # Asignar los nombres de las columnas al DataFrame
    cf_df_clean.columns = all_column_names

    # Resultado limpio y con nombres de columnas
    print(cf_df_clean)
    
    """