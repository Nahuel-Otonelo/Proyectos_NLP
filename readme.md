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

# 💬 Desafío 2: Embeddings de Palabras con Word2Vec (Martín Fierro)

Este proyecto es el segundo desafío de la **materia**, enfocado en la creación y análisis de embeddings de palabras personalizados.

El notebook `Desafio_2.ipynb` implementa la librería `Gensim` para entrenar un modelo **Word2Vec (Skip-gram)**, utilizando un corpus de texto en español.

## 📖 Dataset

Se utilizó un corpus personalizado compuesto por las dos obras de José Hernández: **"El Gaucho Martín Fierro"** y **"La Vuelta de Martín Fierro"**.

Los textos se obtuvieron del sitio `textos.info` en formato `.epub`, se convirtieron a `.txt` con una herramienta online y se limpiaron manualmente para retener únicamente los versos del poema.

## 🛠️ Desafíos y Metodología

### 1. Preprocesamiento del Corpus con NLTK

A diferencia del desafío anterior, se utilizó un enfoque verso por verso (línea por línea). El preprocesamiento se realizó con `NLTK`, incluyendo:

* **Tokenización:** `nltk.word_tokenize` para cada verso.
* **Filtrado de Stop Words:** Se eliminaron las palabras vacías del español (ej. "de", "la", "que") utilizando el corpus `stopwords` de NLTK.
* **Limpieza:** Se filtraron signos de puntuación y números usando `.isalpha()`.

### 2. Entrenamiento de Word2Vec (Gensim)

Se ajustaron los hiperparámetros del modelo `Word2Vec` a un corpus relativamente pequeño (1375 palabras de vocabulario únicas después del filtrado).

### 3. Análisis de Similaridad Semántica

Se utilizó el método `w2v_model.wv.most_similar` para consultar el modelo entrenado y analizar la coherencia semántica de los vectores generados, tanto para palabras similares como no relacionadas.

### 4. Visualización y Clustering (t-SNE)

Se realizó una reducción de dimensionalidad para poder graficar en dos dimensiones las palabras. Luego se interpretaron algunos **de los** resultados:

1.  Se utilizó **t-SNE** (`sklearn.manifold.TSNE`) para "aplastar" los vectores de 100 dimensiones a un espacio 2D.
2.  Se generó un gráfico interactivo con `Plotly Express` para visualizar las 800 palabras más frecuentes.
3.  Se inspeccionó el gráfico para **identificar y analizar clusters semánticos**. Interpretaciones destacadas: