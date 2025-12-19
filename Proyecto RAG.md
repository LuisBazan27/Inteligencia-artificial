# Proyecto – Análisis Semántico y Crisis de Sentido en la Generación Z

Este proyecto analiza textos relacionados con la Generación Z, algoritmos y crisis de sentido
utilizando técnicas de *embeddings*, búsqueda semántica y visualización de datos.
Se emplea un enfoque tipo RAG (Retrieval-Augmented Generation) para encontrar evidencias
filosóficas en los datos.

---

## Carga del dataset desde Excel

Este bloque localiza automáticamente un archivo `.xlsx` en el entorno y lo carga en un DataFrame
para su análisis posterior.

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
