from Analisis import Preprocesamiento

pp = Preprocesamiento("C:/Users/Usuario/OneDrive/Desktop/Practicas-Empresa-2024/Practicas-Empresa-2024/data/Sostenibilidad_tic.csv")
pp.delete("Marca_Temporal")
pp.standarize("Grado", "C:/Users/Usuario/OneDrive/Desktop/Practicas-Empresa-2024/Practicas-Empresa-2024/standard/Grado.txt")
pp.replaceAll("Especialidad", "Matemáticas", "Grado", "ingeniería", "Grado en Ingeniería Matemática")
pp.replaceAll("Especialidad", "Otra", "Grado", "ingeniería", "Grado en Ingeniería (sin especificar)")

