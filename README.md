# Lithuanian LLM Fine-Tuning (Grammar + QA + Poetic Style)

This project generates a Lithuanian instruction-style dataset and fine-tunes small chat LLMs using **LoRA/PEFT** with **4-bit quantization**, aiming to produce grammatically correct Lithuanian answers in a **poetic, classical-inspired style** (inspired by Maironis, without copying any of his texts).

## What was done

1. **Prompt design**
   - Defined system prompts for three tasks in [prompts/system_prompts.py](prompts/system_prompts.py):
     - grammar mistakes + corrections (`system_prompt_grammar`)
     - Lithuanian Q&A with poetic answers (`system_prompt_qa`)
     - simple sentence to poetic rephrasing (`system_prompt_maironis`)
   - Defined matching user prompts in [prompts/user_prompts.py](prompts/user_prompts.py).

2. **Dataset generation**
   - Implemented synthetic dataset generation using `ollama.chat` in [dataset_generation.py](dataset_generation.py).
   - `qwen2.5:32b` model was used for final dataset generation - 1000 examples in total.
   - Outputs training examples in chat format (`messages: [{role, content}, ...]`) via [`to_fine_tuning`](dataset_generation.py).
   - Saved dataset JSON into the [dataset/](dataset/) folder (e.g. `dataset/fine_tuning_dataset_with_32b.json`).

3. **Fine-tuning**
   - Fine-tuned a base model with **TRL SFTTrainer** + **PEFT LoRA** in [llm_fine_tuning.py](llm_fine_tuning.py).
   - Two models - `TinyLlama/TinyLlama-1.1B-Chat-v1.0` and `google/gemma-2b` were fine tuned for comparing purposes.
   - Used 4-bit quantization (BitsAndBytes) and trained adapters, then saved the result to an output directory.

4. **Inference / testing**
   - Loaded the base model + adapter weights and generated responses using the model chat template in:
     - [test_fine_tuned_model.py](test_fine_tuned_model.py)
     - [use_fine_tuned_model.ipynb](use_fine_tuned_model.ipynb)

## Outputs

- Generated datasets: [dataset/](dataset/)
- Fine-tuned adapter directories:
  - [fine_tuned_gemma4_model/](fine_tuned_gemma4_model/)
  - [fine_tuned_tinyllama_model/](fine_tuned_tinyllama_model/)

## How to run (high level)

- Generate dataset: run [dataset_generation.py](dataset_generation.py) (`qwen2.5:32b` needs to be downloaded and run with ollama)
- Fine-tune: run [llm_fine_tuning.py](llm_fine_tuning.py)
- Test: run [test_fine_tuned_model.py](test_fine_tuned_model.py) or open [use_fine_tuned_model.ipynb](use_fine_tuned_model.ipynb)