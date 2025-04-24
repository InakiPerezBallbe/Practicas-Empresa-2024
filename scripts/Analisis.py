import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

class Preprocesamiento:
    
    def __init__(self, link):
        self.link = link
        self.data = pd.read_csv(link)
        self.enc = OrdinalEncoder()

    
    def standarize (self, column_name, link):
        mapping = {}

        with open(link, encoding="utf-8") as f:
            for line in f:
                (key, val) = line.split(":")
                mapping.update([(key, val.strip().split(","))])

        for idvalue, value in enumerate(self.data[column_name]):
            value = value.strip().lower()
            for standard, variations in mapping.items():
                if value in map(str.lower, variations):
                    self.data.loc[idvalue, column_name] = standard.lower()
        
        self.data[column_name] = self.data[column_name].str.lower()

    def delete(self, column_name):
        self.data.drop(columns=[column_name], inplace=True)

    def replaceAll(self, column1, column1_value, column2, column2_value, new_column2_value):
        self.data.loc[(self.data[column1] == column1_value) & (self.data[column2].str.lower() == column2_value), column2] = new_column2_value
        self.data[column2] = self.data[column2].str.lower()
    
    def encode(self):
        self.enc.fit(self.data)
        self.data = pd.DataFrame(self.enc.transform(self.data))

    def decode(self):
        self.data = pd.DataFrame(self.enc.inverse_transform(self.data))

"""
X = data.drop(data.columns[-2], axis=1)
Y = pd.DataFrame(data=data.iloc[:, -2].values, columns=[data.iloc[:, -2].name])
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=42)

"""