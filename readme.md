# 📰 Proyecto 1: Clasificación de Texto con Naive Bayes (Dataset 20 Newsgroups)

Este proyecto es el primer desafío de la matería Procesamiento de Lenguaje Natural (PLN).

El notebook `Desafio_1.ipynb` explora la vectorización de texto (TF-IDF), la similaridad de documentos, la implementación de clasificadores (k-NN y Naive Bayes) y la optimización de hiperparámetros.

## 📊 Dataset

Se utilizó el dataset **20 Newsgroups**, cargado directamente desde `scikit-learn`. Este es un conjunto clásico para la clasificación de texto, compuesto por ~18,000 mensajes de foros distribuidos en 20 categorías temáticas (ej. `rec.autos`, `sci.med`, `talk.politics.misc`).

---

## 🛠️ Desafíos y Metodología

El notebook está dividido en los 4 puntos de la consigna:

### 1. Análisis de Similaridad de Documentos

Se vectorizó el corpus de `train` con `TfidfVectorizer`. Luego, se midió la similaridad coseno entre 5 documentos elegidos al azar y el resto del corpus para analizar la coherencia de las clases de los documentos más similares.

### 2. Clasificador por Prototipos (k-NN)

Se construyó un clasificador 1-NN ("prototipo") asignando la clase del vecino más cercano. Como extensión, se implementó un clasificador **k-NN** completo, probando un rango de $k$ (de 1 a 21) y comparando dos estrategias de votación:

* **Voto Democrático (`weights='uniform'`)**
* **Voto Calificado (`weights='distance'`)**

Se generó un gráfico para comparar el F1-Score de ambas estrategias y encontrar el $k$ óptimo.

### 3. Optimización de Naive Bayes (GridSearch)

El objetivo era maximizar el `f1-score (macro)`:

1.  Se comparó `MultinomialNB` vs. `ComplementNB`, identificando a `ComplementNB` como el modelo superior (probablemente por el desbalance de clases del dataset).
2.  Se implementó un **`Pipeline`** de `scikit-learn` para encadenar el `TfidfVectorizer` y el `ComplementNB`.
3.  Se utilizó **`GridSearchCV`** para encontrar la mejor combinación de hiperparámetros, previniendo el *data leakage* mediante validación cruzada.

### 4. Similaridad de Palabras (Matriz Transpuesta)

Finalmente, se transpuso la matriz TF-IDF (documento-término) para obtener una matriz (término-documento).

* Cada fila se reinterpretó como un **vector de palabra** (un embedding simple).
* Se analizó la similaridad coseno de 5 palabras (`god`, `car`, `president`, etc.) para estudiar las relaciones semánticas y de co-ocurrencia que el modelo fue capaz de capturar.

---

# 📰 Proyecto 1: Clasificación de Texto con Naive Bayes (Dataset 20 Newsgroups)

Este proyecto es el primer desafío de la materia Procesamiento de Lenguaje Natural (PLN).

El notebook `Desafio_1.ipynb` explora la vectorización de texto (TF-IDF), la similaridad de documentos, la implementación de clasificadores (k-NN y Naive Bayes) y la optimización de hiperparámetros.

## 📊 Dataset

Se utilizó el dataset **20 Newsgroups**, cargado directamente desde `scikit-learn`. Este es un conjunto clásico para la clasificación de texto, compuesto por ~18,000 mensajes de foros distribuidos en 20 categorías temáticas (ej. `rec.autos`, `sci.med`, `talk.politics.misc`).

---


# 💬 Desafío 2: Embeddings de Palabras con Word2Vec (Martín Fierro)

Este proyecto es el segundo desafío de la materia, enfocado en la creación y análisis de embeddings de palabras personalizados.

El notebook `Desafio_2.ipynb` implementa la librería `Gensim` para entrenar un modelo **Word2Vec (Skip-gram)**, utilizando un corpus de texto en español.

## 📖 Dataset

Se utilizó un corpus personalizado compuesto por las dos obras de José Hernández: **"El Gaucho Martín Fierro"** y **"La Vuelta de Martín Fierro"**.

Los textos se obtuvieron del sitio `textos.info` en formato `.epub`, se convirtieron a `.txt` con una herramienta online y se limpiaron manualmente para retener únicamente los versos del poema.

## 🛠️ Desafíos y Metodología

### 1. Preprocesamiento del Corpus (El Desafío Clave)

La estrategia para solucionar el inconveniente de versos cortos, fue tratar el poema como un solo bloque de **"prosa"** continuo:

* Se **reemplazaron todos los saltos de línea (`\n`) por espacios**, creando un único string de texto.
* Se utilizó **`text_to_word_sequence` de Keras/TensorFlow** para tokenizar. Esta función se eligió porque:
    1.  Maneja correctamente caracteres especiales (como la `ü` en "vigüela"), a diferencia del filtro `.isalpha()` de NLTK. No reconoció vigüela pero si otras...
    2.  **No elimina las *stop words*** (como "el", "la", "que"), que son cruciales para que Word2Vec aprenda el contexto gramatical que nos faltaba. En una iteracion anterior de este trabajo, habia eliminado stop words y daba valores inconsistentes de similaridad.
* La lista gigante de tokens se dividió en **"fragmentos" (chunks)** de 100 palabras cada uno, que fueron los "documentos" que se pasaron a Gensim.

### 2. Entrenamiento de Word2Vec (Gensim)

Se ajustaron los hiperparámetros del modelo `Word2Vec` (Skip-gram) para un corpus denso pero pequeño. Establecemos:

* `min_count=3`
* `window=5` (para capturar el contexto de la prosa)
* `vector_size=10