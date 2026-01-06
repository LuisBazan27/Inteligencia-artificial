# Tutor Inteligente de Algoritmos con Fine-Tuning y LoRA

## Introducción

Este proyecto consiste en el desarrollo de un tutor inteligente de algoritmos basado en un modelo de lenguaje de gran escala, ajustado mediante fine-tuning supervisado con LoRA (Low-Rank Adaptation). El sistema está diseñado para responder preguntas sobre algoritmos de forma clara, estructurada y paso a paso, simulando el comportamiento de un tutor académico.

El proyecto abarca dos componentes principales: el entrenamiento del modelo a partir de un dataset especializado y la implementación de un sistema interactivo en consola que permite al usuario formular preguntas y recibir explicaciones detalladas.

---

## Objetivo del Proyecto

Desarrollar un modelo de lenguaje especializado que sea capaz de explicar conceptos de algoritmos de manera didáctica, proporcionar respuestas paso a paso, adaptarse a un contexto educativo y funcionar como un tutor interactivo en tiempo real.

---

## Modelo Base

Se utilizó como modelo base el modelo TinyLlama/TinyLlama-1.1B-Chat-v1.0. Este modelo fue seleccionado por su bajo costo computacional y su compatibilidad con técnicas de fine-tuning ligero, permitiendo su ajuste sin requerir grandes recursos de hardware.

---

## Dataset

El entrenamiento se realizó utilizando un dataset propio en formato JSONL ubicado en:

data/processed/tutor_algoritmos_clean.jsonl

Cada registro del dataset contiene:
- instruction: pregunta o consigna relacionada con algoritmos.
- output: respuesta esperada redactada con enfoque pedagógico.

El dataset fue previamente limpiado y normalizado para asegurar coherencia semántica y calidad en las respuestas generadas.

---

## Metodología de Entrenamiento

### Fine-Tuning con LoRA

Para el ajuste del modelo se empleó la técnica LoRA (Low-Rank Adaptation), lo que permitió reducir el número de parámetros entrenables, disminuir el consumo de memoria y mantener congelado el modelo base, adaptando únicamente capas específicas.

Configuración de LoRA:
- Rank (r): 8  
- Alpha: 16  
- Dropout: 0.05  
- Tipo de tarea: Causal Language Modeling  

---

### Formato de los Ejemplos

Los ejemplos de entrenamiento se estructuraron de la siguiente forma:

### Instrucción:
[pregunta]

### Respuesta:
[respuesta del tutor]

Este formato facilita que el modelo aprenda la separación entre la pregunta del estudiante y la explicación del tutor.

---

### Parámetros de Entrenamiento

- Batch size por dispositivo: 1  
- Gradient accumulation steps: 4  
- Número de épocas: 3  
- Learning rate: 2e-4  
- Precisión: float32  
- Estrategia de guardado: por época  

El entrenamiento se realizó mediante la clase SFTTrainer de la librería TRL.

---

## Arquitectura del Sistema

El sistema se divide en dos flujos principales:

Entrenamiento del modelo:
- Carga del dataset.
- Tokenización de los ejemplos.
- Aplicación de LoRA sobre el modelo base.
- Entrenamiento supervisado.
- Guardado del modelo ajustado.

Ejecución del tutor:
- Carga del modelo entrenado.
- Recepción de preguntas por consola.
- Generación de respuestas con muestreo controlado.
- Presentación de la respuesta al usuario.

---

## Sistema de Inferencia

El tutor funciona como una aplicación interactiva en consola. Para cada pregunta del usuario, el sistema construye un prompt con rol explícito de tutor experto, tokeniza la entrada, genera la respuesta usando el modelo entrenado y devuelve una explicación clara y estructurada.

Parámetros de generación:
- Máximo de tokens nuevos: 300  
- Sampling activado  
- Temperature: 0.7  
- Top-p: 0.9  

Estos parámetros permiten un equilibrio entre coherencia y variabilidad en las respuestas.

---

## Ejecución del Proyecto

Para iniciar el tutor, ejecutar:

python main.py

El usuario puede escribir preguntas libremente sobre algoritmos. Para finalizar la sesión, escribir:

salir

---

## Resultados Esperados

El sistema es capaz de explicar algoritmos paso a paso, utilizar un lenguaje claro y pedagógico, mantener coherencia temática en sus respuestas y adaptarse tanto a preguntas conceptuales como procedimentales. El comportamiento del modelo refleja el conocimiento adquirido durante el fine-tuning con el dataset especializado.

---

## Tecnologías Utilizadas

- Python  
- PyTorch  
- Hugging Face Transformers  
- Datasets  
- TRL (SFTTrainer)  
- PEFT (LoRA)  

---

## Estructura del Proyecto

/data  
  /processed  
    tutor_algoritmos_clean.jsonl  
/models  
  /tutor_algoritmos  
/scripts  
  train.py  
  main.py  
README.md  

---

## Conclusión

Este proyecto demuestra que es posible construir un tutor inteligente especializado mediante fine-tuning ligero con LoRA, obteniendo un modelo capaz de brindar explicaciones claras y estructuradas sobre algoritmos. La combinación de un dataset curado, un modelo eficiente y una estrategia de entrenamiento adecuada permite desarrollar asistentes educativos funcionales sin requerir grandes infraestructuras computacionales.


```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models/tutor_algoritmos"

def main():
    print("Tutor Inteligente de Algoritmos")
    print("Escribe 'salir' para terminar\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32
    )

    model.eval()

    while True:
        user_input = input("Tú: ")

        if user_input.lower() in ["salir", "exit", "quit"]:
            print("Hasta luego")
            break

        prompt = f"""Eres un tutor experto en algoritmos.
Explica paso a paso y con claridad.

Pregunta del estudiante:
{user_input}

Respuesta del tutor:
"""

        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        response = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        print("\nTutor:\n")
        print(response.replace(prompt, "").strip())
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    main()
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "data/processed/tutor_algoritmos_clean.jsonl"
OUTPUT_DIR = "models/tutor_algoritmos"

# Cargar dataset
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# Tokenizador y modelo base
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Configuración LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Formato de los ejemplos de entrenamiento
def format_example(example):
    return f"### Instrucción:\n{example['instruction']}\n\n### Respuesta:\n{example['output']}"

# Argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none"
)

# Entrenador
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=format_example,
    args=training_args
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
