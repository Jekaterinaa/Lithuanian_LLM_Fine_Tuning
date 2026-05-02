import os
os.environ["PYTHONUTF8"] = "1"

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer


print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("bf16 support:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else "N/A")

# --- Model ---
# needs verification now
# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_name = "unsloth/Llama-3.2-3B-Instruct"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    dtype=torch.bfloat16,
)
model.config.use_cache = False  # required for gradient checkpointing

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- Dataset ---
# Adjust path/format to your dataset
dataset = load_dataset("json", data_files="dataset/fine_tuning_dataset_with_32b.json", split="train")

# --- LoRA Config ---
peft_config = LoraConfig(
    r=8,              # rank dimension (4-32 typical; 8 is a solid default)
    lora_alpha=16,     # scaling factor (commonly 2x r)
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# --- Training Arguments ---
output_dir = "./llama3-finetuned"
max_seq_length = 1024

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    bf16=False,
    fp16=False,
    use_cpu=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
)

# --- Trainer ---
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    processing_class=tokenizer,
)

# --- Train ---
trainer.train()

# --- Save ---
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")
