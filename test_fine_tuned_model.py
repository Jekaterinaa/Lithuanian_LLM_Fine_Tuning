# Load the fine-tuned model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel
import torch


fine_tuned_model_path = "fine_tuned_tinyllama_model"
# Define the base model name (same as used for training)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Define the quantization config (same as used for training)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 1. Load the original base model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    dtype=torch.bfloat16,
)

# 2. Load the PEFT adapter weights and apply them
# Assuming output_dir contains the adapter weights (adapter_model.safetensors, adapter_config.json)
loaded_model = PeftModel.from_pretrained(
    base_model,
    fine_tuned_model_path,
    dtype=torch.float16
)

# Optionally, merge the LoRA weights into the base model
# This can be useful for deployment, but increases memory usage if not needed
# loaded_model = loaded_model.merge_and_unload()

# Re-initialize the tokenizer
loaded_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
loaded_tokenizer.pad_token = loaded_tokenizer.eos_token # Ensure tokenizer settings are consistent
loaded_tokenizer.padding_side = "right"

print("Model and tokenizer loaded successfully!")
print("Note: The model is loaded with 4-bit quantization and PEFT adapters applied.")


def generate(prompt, max_new_tokens=100):
    messages = [
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = loaded_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = loaded_tokenizer(formatted_prompt, return_tensors="pt").to(loaded_model.device)

    with torch.no_grad():
        outputs = loaded_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return loaded_tokenizer.decode(new_tokens, skip_special_tokens=True)


prompt = "kas yra gilusis mokymasis ir kaip jis veikia?"
print(generate(prompt))

# # Create a text generation pipeline
# pipeline = pipeline(
#     "text-generation",
#     model=loaded_model,
#     tokenizer=loaded_tokenizer,
#     dtype=torch.bfloat16, # Ensure data type matches your training (bf16)
#     device=0 # Use GPU if available
# )

# # Define your prompt
# prompt = "Kas yra gilusis mokymasis ir kaip jis veikia?"

# # Generate text
# sequences = pipeline(
#     prompt,
#     max_new_tokens=100,
#     do_sample=True,
#     top_k=50,
#     top_p=0.95,
#     num_return_sequences=1,
# )

# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")