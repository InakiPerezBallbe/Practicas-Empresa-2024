import pandas as pd
import dice_ml
import lime
import lime.lime_tabular
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC

class Preprocessing:
    
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
        encoders = {}

        with open(link, encoding="utf-8") as f:
            enc_type = None

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
                        encoders[col] = le
                    elif enc_type == "OE":
                        categories = None
                        if ":" in col:
                            col, cat_str = col.split(":")
                            categories = [cat_str.split(";")]
                        oe = OrdinalEncoder(categories=categories) if categories else OrdinalEncoder()
                        self.data[col] = oe.fit_transform(self.data[[col]])
                        encoders[col] = oe 
                    elif enc_type == "OHE":
                        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                        transformed = ohe.fit_transform(self.data[[col]])
                        columns = ohe.get_feature_names_out([col])
                        df_ohe = pd.DataFrame(transformed, columns=columns, index=self.data.index)
                        self.data = self.data.drop(columns=[col]).join(df_ohe)
                        encoders[col] = ohe
        
        return encoders

class Clasification:

    def __init__ (self, data, target, test_size):
        Y = data[target]
        X = data.drop(target, axis=1)
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(X, Y, test_size=test_size, random_state=42)
        smotenc = SMOTENC(categorical_features=list(range(0, len(data.columns)-2)), random_state=42)
        self.xtrain, self.ytrain = smotenc.fit_resample(self.xtrain, self.ytrain)
        self.xtrain = self.xtrain.round(decimals=0)
        self.list_models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'SVC': SVC(probability=True, random_state=42),
            'Naive Bayes': GaussianNB(),
            'MLP Classifier': MLPClassifier(max_iter=1000, random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
        }

    def trainEvaluate(self):
        records = []
        
        for name, model in self.list_models.items():
                model.fit(self.xtrain, self.ytrain)
                ypred = model.predict(self.xtest)
                
                record = {
                    'Model': name,
                    'Accuracy': accuracy_score(self.ytest, ypred),
                    'Precision': precision_score(self.ytest, ypred, average='weighted'),
                    'Recall': recall_score(self.ytest, ypred, average='weighted'),
                    'F1-Score': f1_score(self.ytest, ypred, average='weighted')
                }
                records.append(record)
            
        return pd.DataFrame(records).sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
    
    def train(self, models_selected = []):
        estimators = []

        for name, model in self.list_models.items():
            if name in models_selected:
                estimators.append((name, model))

        self.model = VotingClassifier(estimators=estimators, voting='soft')
        self.model.fit(self.xtrain, self.ytrain)
        ypred = self.model.predict(self.xtest)
        record = {
                'Model': "VotingClassifier",
                'Accuracy': accuracy_score(self.ytest, ypred),
                'Precision': precision_score(self.ytest, ypred, average='weighted'),
                'Recall': recall_score(self.ytest, ypred, average='weighted'),
                'F1-Score': f1_score(self.ytest, ypred, average='weighted')
            }

        return pd.DataFrame([record])

    def predict(self, row):
        ypred = self.model.predict(row)
        yprob = self.model.predict_proba(row)
        row_pred = pd.concat([row, pd.DataFrame(ypred, columns=[self.ytrain.name])], axis=1)

        return row_pred, (yprob[0][ypred] * 100).round(2)

class Counterfactual:

    def __init__(self, model, data, target):
        self.model = model
        self.data = data
        self.target = target
        dmodel = dice_ml.Model(model=model, backend="sklearn")
        d = dice_ml.Data(dataframe=data, continuous_features=[], outcome_name=target)
        self.exp = dice_ml.Dice(d, dmodel, method="random")

    def counterfac(self, row, encoders):
        cf_row = row.iloc[:, row.columns != self.target]
        dice_exp = self.exp.generate_counterfactuals(cf_row, total_CFs=4, desired_class="opposite", random_seed=42)
        cf_data = dice_exp.cf_examples_list[0].final_cfs_df
        cf_data = self.__decode(cf_data, encoders)
        cf_row = self.__decode(row, encoders)

        for col in cf_data.columns:
            if (cf_data[col] == cf_row.iloc[0][col]).all():
                cf_data.drop(columns=col, inplace=True)
                cf_row.drop(columns=col, inplace=True)

        print(cf_row)
        print(cf_data)

    def __decode(self, data, encoders):
        for col, encoder in encoders.items():
                    if isinstance(encoder, LabelEncoder):
                        data[col] = data[col].astype('int64')
                        data[col] = encoder.inverse_transform(data[col])

                    elif isinstance(encoder, OrdinalEncoder):
                        data[col] = encoder.inverse_transform(data[[col]]).ravel()

                    elif isinstance(encoder, OneHotEncoder):
                        ohe_cols = encoder.get_feature_names_out([col])
                        data[col] = encoder.inverse_transform(data[ohe_cols]).ravel()
                        data = data.drop(columns=ohe_cols)
        
        cols = list(data.columns)
        cols[cols.index(self.target)], cols[-1] = cols[-1], cols[cols.index(self.target)]
        return data[cols]
