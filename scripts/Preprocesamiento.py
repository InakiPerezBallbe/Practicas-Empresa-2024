import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("C:/Users/Usuario/OneDrive/Desktop/Practicas-Empresa-2024/Practicas-Empresa-2024/data/Sostenibilidad_tic.csv")

#Quitar la Marca Temporal
data.drop(columns=["Marca_Temporal"], inplace=True)

degree_mapping = {
    "Grado en Filología Hispánica": ["Filología Hispánica", "Grado Filología Hispánica", "Filología Hispánica "],
    "Grado en Química": ["grado química", "QUIMICA"],
    "Grado en Ingeniería Química": ["Ingenieria Quimica"],
    "Grado en Administración y Dirección de Empresas": ["ADE", "Administración y Dirección de empresas", "Administración y Dirección de Empresas", "ADMINISTRACIÓN Y DIRECCIÓN DE EMPRESAS", "G.ADE", "ADMINISTRACION Y DIRECCIÓN DE EMPRESAS" ],
    "Licenciatura en Administración y Dirección de Empresas": ["LADE"],
    "Grado en Matemáticas": ["Grado en matematicas", "Matemáticas"],
    "Grado en Ingeniería de Telecomunicaciones": ["Ingeniería de Telecomunicación", "GRADO TELECOMUNICACIONES"],
    "Doble Grado en Estudios Ingleses y Filología Hispánica": ["Doble Grado en Estudios Ingleses y Filología Hispánica", "Doble Grado F. Hispánica y E. Ingleses", "DG Estudios ingleses y Filología hispánica", "Doble Grado en Estudios Ingleses y Filología Hispánica"],
    "Grado en Historia": ["historia", "HISTORIA"],
    "Grado en Biotecnología": ["Biotecnología"],
    "Grado en Estudios Ingleses": ["Estudios Ingleses"],
    "Grado en Geología": ["en Geología"],
    "Grado en Bellas Artes": ["Bellas Artes"],
    "Grado en Periodismo": ["Periodismo"],
    "Grado en Finanzas y Contabilidad": ["FINANZAS Y CONTABILIDAD"],
    "Grado en Ingeniería Informática": ["Ingeniería Informática"],
    "Grado en Ciencias de la Actividad Física y el Deporte": ["Ciencias de la actividad fisica y el deporte", "Ciencias de la Actividad Física y del Deporte", "Ciencia de la actividad fisica y del deporte"],
    "Grado en Ingeniería Agrícola": ["Ingeniería Agrícola", "INGENIERIA AGRÓNOMA"],
    "Grado en Ingeniería Electrónica Industrial": ["ingenieria electronica industrial", "Grado en Ingeniería Electrónica industrial"],
    "Grado en Ingeniería Técnica Industrial": ["INGENIERO TÉCNICO INDUSTRIAL"],
    "Grado en Diseño de Interiores": ["Diseño de Interiores"],
    "Grado en Humanidades": ["Humanidades"],
    "Grado en Farmacia": ["Farmacia"],
    "Grado en Biología": ["Biología"],
    "Grado en Arquitectura": ["Arquitectura"],
    "Grado en Geografía y Gestión del Territorio": ["Geografía y gestión del territorio"],
    "Grado en Ciencias Políticas y de la Administración": ["CIENCIAS POLITICAS Y DE LA ADMINISTRACIÓN"],
    "Grado en Conservación y Restauración de Bienes Culturales": ["Conservacion y restauracion de bienes culturales"],
    "Grado en Ingeniería Mecánica": ["Ingeniería Mecánica", "MECANICA", "Ingeniería mécnaica"]
}

# Función para estandarizar
def standardize_degree(value):
    # Para eliminar espacios en blanco al inicio y al final
    value = value.strip().lower()
    for standard, variations in degree_mapping.items():
        if value in map(str.lower, variations):  # Compara todo en minúsculas
            return standard.lower()
    return value  # Si no se encuentra en el diccionario, dejar el original

data["Grado"] = data["Grado"].apply(standardize_degree).str.lower()

# Si el grado es "ingeniería" y la especialidad es "Matemáticas", asignamos un grado más específico
data.loc[(data["Especialidad"] == "Matemáticas") & (data["Grado"].str.lower() == "ingeniería"), "Grado"] = "Grado en Ingeniería Matemática"
# Si el grado es "ingeniería" y la especialidad es "Otro"
data.loc[(data["Especialidad"] == "Otra") & (data["Grado"].str.lower() == "ingeniería"), "Grado"] = "Grado en Ingeniería (sin especificar)"
data["Grado"] = data["Grado"].str.lower()

"X = data.iloc[:, 0:-1]"
X = data.drop(data.columns[-2], axis=1)
Y = data.iloc[:, -2]
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=42)