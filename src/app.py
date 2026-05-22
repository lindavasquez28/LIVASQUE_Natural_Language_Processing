
# # LINDA VASQUEZ Natural Language Processing project

import pandas as pd

enlace = "https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv"
datos = pd.read_csv(enlace)

# ## Procesamiento de datos
# para spam, 0 para no spam
datos["es_spam"] = datos["is_spam"].apply(lambda x: 1 if x else 0)

# eliminar filas repetidas
datos = datos.drop_duplicates()
datos = datos.reset_index(drop=True) # Reseteamos el índice para que quede ordenado

print("Cantidad de correos Spam:", len(datos[datos["es_spam"] == 1]))
print("Cantidad de correos Normales:", len(datos[datos["es_spam"] == 0]))

import re
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# herramientas de NLTK
download("wordnet")
download("stopwords")

lematizador = WordNetLemmatizer()
palabras_vacias = stopwords.words("english") # Palabras que no aportan significado [cite: 31]

def limpiar_y_lematizar(texto):
    # solo letras minúsculas
    texto = re.sub(r'[^a-z ]', " ", texto.lower())
    # sin espacios extra
    texto = re.sub(r'\s+', " ", texto)
    
    # Tokenizacion
    palabras = texto.split()
    
    tokens = [lematizador.lemmatize(p) for p in palabras]
    tokens = [p for p in tokens if p not in palabras_vacias and len(p) > 3]
    
    return tokens

datos["url_limpia"] = datos["url"].apply(limpiar_y_lematizar)
datos.head()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

textos = [" ".join(lista) for lista in datos["url_limpia"]]

# vectorizacion limitando a 5000 palabras mas importantes
vectorizador = TfidfVectorizer(max_features=5000, max_df=0.8, min_df=5)
X = vectorizador.fit_transform(textos).toarray()
y = datos["es_spam"]

# 80% para entrenar y 20% para probar
X_entrena, X_prueba, y_entrena, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

# ## SVM
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

modelo_base = SVC(kernel="linear", random_state=42)

modelo_base.fit(X_entrena, y_entrena)

predicciones_base = modelo_base.predict(X_prueba)
precision_base = accuracy_score(y_prueba, predicciones_base)

print(f"Precision del modelo base: {precision_base}")

# ## Optimización

from sklearn.model_selection import GridSearchCV

parametros = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "poly", "rbf"]
}

busqueda = GridSearchCV(modelo_base, parametros, cv=3) # cv=3 es más rápido que 5
busqueda.fit(X_entrena, y_entrena)

print(f"Los mejores parámetros encontrados son: {busqueda.best_params_}")

modelo_ganador = busqueda.best_estimator_
predicciones_optimizadas = modelo_ganador.predict(X_prueba)
precision_optimizada = accuracy_score(y_prueba, predicciones_optimizadas)

print(f"Precision del modelo optimizado: {precision_optimizada}")

# Guardado
import pickle

ruta_guardado = "modelo_svm_spam_optimizado.sav"
pickle.dump(modelo_ganador, open(ruta_guardado, "wb"))

print("Modelo guardado, vea el src")
