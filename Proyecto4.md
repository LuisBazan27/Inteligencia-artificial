# Proyecto 4 – Tutor Inteligente de Algoritmos con LLM

Este proyecto implementa un tutor inteligente especializado en algoritmos utilizando modelos de lenguaje.
Incluye dos partes:
- Uso del modelo entrenado como tutor interactivo
- Entrenamiento del modelo mediante fine-tuning con LoRA

---

## Tutor interactivo de algoritmos

Este script carga un modelo previamente entrenado y permite interactuar con él desde la terminal,
respondiendo preguntas sobre algoritmos de forma clara y paso a paso.

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
