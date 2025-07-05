# Poetry Generation using Generative AI

## Overview

This project explores the use of Generative AI for creative text generation, specifically focusing on English poetry. We fine-tuned a GPT-2 language model on a curated dataset of classic poems and compared its outputs with those from a state-of-the-art, locally run Llama-based model (`bartowski/Llama-3.2-1B-Instruct-GGUF`) using Ollama. The project features an interactive web GUI for generating poems based on user prompts.

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Usage](#usage)
- [Model Details](#model-details)
- [Testing & Evaluation](#testing--evaluation)
- [Future Work](#future-work)
- [Sample Outputs](#sample-outputs)
- [References](#references)


## Features

- **Fine-tuned GPT-2 Model:** Custom-trained on a dataset of classic English poems for stylistic and creative output.
- **Pretrained Llama-3.2-1B (GGUF) Model:** Used via Ollama for comparison and benchmarking.
- **Interactive Web GUI:** Generate poems by entering prompts and adjusting creativity settings.
- **CLI Script:** Command-line interface for quick poem generation.
- **Comprehensive Testing:** Manual, automated, and user testing for robustness and quality.

## Setup & Installation

### 1. Clone the Repository

```sh
git clone 
cd poetry-gen-ai
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

**Requirements include:**  
- `transformers`
- `datasets`
- `torch`
- `gradio`
- `ollama`

### 3. Set Up Ollama (for GGUF Model)

- Download and install [Ollama](https://ollama.com/download) for your OS.
- Pull the Llama model:
  ```sh
  ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
  ```
- Start the Ollama server:
  ```sh
  ollama serve
  ```

### 4. Prepare Poetry Dataset (for Fine-tuning)

- Place cleaned `.txt` files (one per poet) in the `data/` directory.
- Combine and preprocess as needed.

## How to Run

### 1. **Fine-Tune GPT-2 Model**

If you want to retrain:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Load and preprocess your dataset...
# (See src/generate_poetry.py for details)

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# Set pad_token if needed: tokenizer.pad_token = tokenizer.eos_token

training_args = TrainingArguments(
    output_dir="./poetry-gpt2-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=1000,
    save_total_limit=1,
    logging_steps=100,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)
trainer.train()
model.save_pretrained("./poetry-gpt2-finetuned")
tokenizer.save_pretrained("./poetry-gpt2-finetuned")
```

Or use the provided pretrained model in `poetry-gpt2-finetuned/`.

### 2. **Run the Gradio Web GUI**

```sh
python src/poetry_gui.py         # For GPT-2 model
python src/ollama_poetry_gui.py  # For Ollama GGUF model
```

### 3. **Run from the Command Line**

```sh
python src/generate_poetry.py
```

## Usage

- **Web GUI:**  
  - Enter a prompt (e.g., "Write a poem about the sea").
  - Adjust creativity (temperature) and length (max tokens).
  - Click "Generate Poem" to see the result.

- **CLI:**  
  - Run the script and follow the prompts in your terminal.

## Model Details

- **Fine-tuned GPT-2:**  
  - Trained on a curated dataset of poems by classic English poets.
  - Learns stylistic elements, rhyme, and structure.

- **bartowski/Llama-3.2-1B-Instruct-GGUF:**  
  - Pulled and run locally with Ollama.
  - Used as a baseline for comparison.

## Testing & Evaluation

- **Unit & Integration Testing:**  
  - All core modules and interfaces tested for correctness and reliability.
- **User Testing:**  
  - Feedback collected on poem quality, interface usability, and overall experience.
- **Performance:**  
  - Measured for prompt response time and output fluency.

## Future Work

- Fine-tune on larger and more diverse poetry datasets.
- Add support for structured poetic forms (sonnet, haiku, etc.).
- Enable collaborative poem creation and user feedback loops.
- Expand to multi-language and multimodal (image-to-poem) generation.

## Sample Outputs

**Prompt:**  
> Write a poem about a quiet morning in the woods.

**GPT-2 Output:**  
> Mist curls softly above the moss,  
> Sunlight drips through waking trees,  
> A hush of dew on silent leaves,  
> The world begins with gentle ease.

**Bartowski GGUF Model:**  
> In the hush of dawn’s embrace,  
> The forest wakes with gentle grace,  
> Birds sing softly, shadows fade,  
> Morning’s peace in woodland shade.

## References

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Ollama](https://ollama.com/)
- [Project Gutenberg Poetry Collections](https://www.gutenberg.org/)
- [Bartowski/Llama-3.2-1B-Instruct-GGUF on Hugging Face](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF)

**For questions, suggestions, or contributions, please open an issue or pull request!**

