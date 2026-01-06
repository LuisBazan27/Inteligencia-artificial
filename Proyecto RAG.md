# Proyecto de Análisis de Datos mediante RAG

##  Introducción

La presente investigación tiene como objetivo analizar, mediante un enfoque basado en **Retrieval-Augmented Generation (RAG)**, dos problemáticas filosóficas contemporáneas estrechamente vinculadas al contexto digital actual: la posible **crisis de sentido en la Generación Z** derivada de la hiperconectividad, y el **papel de los algoritmos digitales y la inteligencia artificial** en la construcción de la identidad y la autonomía.

El estudio adopta un enfoque cualitativo apoyado por técnicas de **análisis semántico asistido por inteligencia artificial**, articulando evidencia textual con marcos filosóficos contemporáneos. Para ello, se implementó un sistema RAG ligero basado en **embeddings semánticos multilingües** y recuperación de evidencia mediante **similitud coseno**, permitiendo interpretar discursos digitales desde una perspectiva crítica.

---

##  Pregunta de Investigación

**¿Cómo se manifiesta una posible crisis de sentido en la Generación Z y qué papel desempeñan los algoritmos digitales y la inteligencia artificial en la construcción de su identidad y autonomía en la era de la hiperconectividad?**

---

##  Hipótesis de Investigación

- La Generación Z presenta indicios de una crisis de sentido, expresada mediante discursos digitales que reflejan vacío existencial y agotamiento emocional.
- La hiperconectividad influye en la construcción de una identidad fragmentada y cambiante, coherente con la noción de *identidad líquida* de Bauman.
- Los algoritmos de recomendación influyen indirectamente en gustos, hábitos y decisiones personales.
- Predominan emociones negativas como ansiedad, cansancio y frustración en discursos relacionados con productividad y visibilidad digital.
- La autonomía percibida es ambigua, al encontrarse mediada por sistemas algorítmicos.
- Se observa un rechazo a los metarrelatos tradicionales, sustituidos por narrativas individuales y efímeras.
- La cultura del rendimiento descrita por Byung-Chul Han se manifiesta en prácticas de autoexplotación digital.

Estas hipótesis no buscan validación causal, sino **exploración interpretativa mediante análisis semántico asistido por un sistema RAG**.

---

##  Marco Teórico-Filosófico

### Crisis de sentido en la Generación Z
- **Sartre / Camus**: vacío existencial.
- **Lyotard**: crisis de los metarrelatos.
- **Bauman**: identidad líquida.
- **Byung-Chul Han**: cultura del rendimiento y burnout.

### Tecnología, IA y Autonomía
- **Foucault**: vigilancia y control algorítmico.
- **Heidegger**: la tecnología como forma de desocultamiento.
- **Habermas**: debilitamiento del espacio público digital.

---

##  Metodología

###  Construcción del Dataset

Se creó un **dataset propio en formato Excel**, compuesto por textos relacionados con:

- Generación Z y crisis de sentido  
- Identidad digital  
- Autonomía y algoritmos  
- Cultura del rendimiento y burnout  

Cada registro contiene:
- Texto
- Tema filosófico
- Sentimiento (positivo, neutral, negativo)
- Métricas de impacto (likes y reposts)

---

###  Limpieza y Preparación

- Eliminación de valores nulos
- Normalización básica de texto
- Conservación del contenido emocional y discursivo

---

###  Generación de Embeddings

Los textos fueron transformados en embeddings utilizando el modelo:

**`paraphrase-multilingual-MiniLM-L12-v2`**

Este modelo permite capturar relaciones semánticas profundas entre conceptos filosóficos y emociones.

---

###  Recuperación de Evidencia (Vector Search)

- Los embeddings se almacenan en memoria dentro de un DataFrame
- La búsqueda se realiza mediante **similitud coseno**
- Se recuperan los textos más relevantes para cada consulta

Este proceso cumple con el principio fundamental del enfoque **RAG**: *recuperar evidencia antes de interpretar*.

---

##  Pipeline del Sistema RAG

1. Formulación de la consulta
2. Generación del embedding de la consulta
3. Cálculo de similitud coseno
4. Recuperación de textos relevantes
5. Interpretación filosófica de la evidencia

---

## Resultados del Análisis

### Influencia de los algoritmos en la autonomía
Los discursos reflejan una pérdida de control percibida, con decisiones mediadas por recomendaciones algorítmicas.

### Cultura del rendimiento y burnout
Predominan expresiones de cansancio, autoexigencia y agotamiento emocional.

### Identidad líquida
Se identifican patrones de cambio constante en valores, intereses y autoimagen.

---

##  Análisis Exploratorio de Datos

- Gráfica de distribución de sentimientos por eje temático
- Nube de palabras centrada en la crisis de sentido

Estas visualizaciones refuerzan empíricamente los hallazgos del sistema RAG.

---

##  Discusión Filosófica

Los resultados reflejan una manifestación contemporánea del vacío existencial descrito por Sartre y Camus, intensificado por entornos digitales. La identidad líquida de Bauman y la cultura del rendimiento de Byung-Chul Han se ven reforzadas por discursos de agotamiento y autoexplotación. Desde Foucault, se evidencia una forma sutil de control algorítmico que condiciona la autonomía juvenil.

---

##  Conclusiones

El sistema RAG implementado sugiere la existencia de indicios consistentes de una crisis de sentido en la Generación Z. Los discursos analizados muestran una identidad fragmentada, emociones negativas recurrentes y una autonomía condicionada por dinámicas algorítmicas. La integración de análisis semántico con reflexión filosófica demuestra el potencial del enfoque RAG para estudiar fenómenos sociotecnológicos contemporáneos.

---

## Tecnologías Utilizadas

- Python
- Pandas
- SentenceTransformers
- Scikit-learn
- Matplotlib / Seaborn
- WordCloud

```python
import pandas as pd
import os

archivo_excel = ""
for f in os.listdir('/content/'):
    if f.endswith('.xlsx'):
        archivo_excel = f'/content/{f}'
        break

if archivo_excel:
    try:
        df = pd.read_excel(archivo_excel)
        print(f"✅ ¡Éxito! Archivo Excel cargado: {archivo_excel}")
        print(f"Total de registros: {len(df)}")
        print("\nColumnas encontradas:")
        print(df.columns.tolist())

        display(df.head())
    except Exception as e:
        print(f"Error al leer el Excel: {e}")
else:
    print("No encontré ningún archivo .xlsx en la carpeta /content. Por favor, súbelo de nuevo.")

!pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np

# Modelo multilingüe
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Generación de embeddings
print("Generando vectores semánticos gratuitos...")
textos = df['texto'].fillna("").tolist()
embeddings = model.encode(textos, show_progress_bar=True)

df['embedding'] = list(embeddings)
print("Terminado")
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def buscar_evidencia(pregunta, n=5):

    query_vector = model.encode([pregunta])

    similitudes = cosine_similarity(
        query_vector,
        np.array(df['embedding'].tolist())
    )[0]

    indices_top = similitudes.argsort()[-n:][::-1]
    resultados = df.iloc[indices_top].copy()

    print(f"--- 🔍 EVIDENCIAS PARA: '{pregunta}' ---\n")
    for i, r in resultados.iterrows():
        print(f"📌 TEMA: {r['tema']} | SENTIMIENTO: {r['sentimiento']}")
        print(f"💬 TEXTO: {r['texto']}")
        print(f"📈 IMPACTO: {r['likes']} likes, {r['reposts']} reposts")
        print("-" * 50)

    return resultados
print("TEST 1: FOUCAULT / AUTONOMÍA")
res1 = buscar_evidencia("¿Cómo influye el algoritmo en la falta de autonomía y control?")

print("\nTEST 2: BYUNG-CHUL HAN / BURNOUT")
res2 = buscar_evidencia("presión por el rendimiento, cansancio y autoexplotación digital")

print("\nTEST 3: BAUMAN / IDENTIDAD LÍQUIDA")
res3 = buscar_evidencia("identidad cambiante, falta de compromiso y cultura de lo efímero")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 7))

grafica = sns.countplot(
    data=df,
    x='tema',
    hue='sentimiento',
    palette={
        'negativo': '#e74c3c',
        'neutral': '#95a5a6',
        'positivo': '#2ecc71'
    }
)

plt.title(
    'Análisis de Sentimientos: Generación Z y Era Digital',
    fontsize=18,
    pad=20,
    fontweight='bold'
)
plt.xlabel('Ejes de Análisis Filosófico', fontsize=12, fontweight='bold')
plt.ylabel('Cantidad de Publicaciones / Testimonios', fontsize=12, fontweight='bold')
plt.xticks(rotation=15, ha='right')

plt.legend(title='Estado de Ánimo', loc='upper right')
plt.tight_layout()
plt.show()
from wordcloud import WordCloud
import matplotlib.pyplot as plt

texto_crisis = " ".join(
    df[df['tema'] == 'Generación Z y crisis de sentido']['texto']
)

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='magma'
).generate(texto_crisis)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Conceptos Clave: Crisis de Sentido en la Gen Z')
plt.show()
