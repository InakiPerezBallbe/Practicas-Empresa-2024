import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
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
        Y = data[target]
        X = data.drop(target, axis=1)
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(X, Y, test_size=test_size, random_state=42)

    


    
        

"""
X = data.drop(data.columns[-2], axis=1)
Y = pd.DataFrame(data=data.iloc[:, -2].values, columns=[data.iloc[:, -2].name])
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=42)

"""