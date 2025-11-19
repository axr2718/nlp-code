import json
import requests
import os
import torch
import gc
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
MODEL_ID = "tiiuae/Falcon3-Mamba-7B-Base"
OUTPUT_DIR = "./mamba_cl_project"
SEQ_LENGTH = 1024    # Matches Paper
LORA_RANK = 8        # Matches Paper
LORA_ALPHA = 16      # Matches Paper
LORA_DROPOUT = 0.1   # Matches Paper
LR = 2e-4            # Matches Paper
EPOCHS_PER_TASK = 3  # Matches Paper

# The specific raw URLs for Order 1 (QA -> QG -> SA)
TASK_URLS = {
    "Task1_QA": "https://raw.githubusercontent.com/allenai/natural-instructions/master/tasks/task024_cosmosqa_answer_generation.json",
    "Task2_QG": "https://raw.githubusercontent.com/allenai/natural-instructions/master/tasks/task074_squad1.1_question_generation.json",
    "Task3_SA": "https://raw.githubusercontent.com/allenai/natural-instructions/master/tasks/task1312_amazonreview_polarity_classification.json"
}

# ==========================================
# 2. DATA ENGINEERING (PHASE 1)
# ==========================================
class SuperNILoader:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def download_and_parse(self, task_name, url, sample_size=2000, eval_size=500):
        print(f"Downloading {task_name}...")
        response = requests.get(url)
        data = response.json()

        # Extract definition (instruction)
        definition = data["Definition"][0] # Usually a list, take first

        # Extract instances
        instances = data["Instances"]
        formatted_data = []

        print(f"  Parsing {len(instances)} instances...")
        for item in instances:
            # Format: Definition + Input -> Output
            # We use a clear separator.
            input_text = f"Definition: {definition}\n\nInput: {item['input']}\n\nOutput:"
            output_text = f" {item['output'][0]}" # Space before output is important

            formatted_data.append({
                "text": input_text + output_text, # For training (Causal LM)
                "input_prompt": input_text,       # For eval (Generation)
                "ground_truth": item['output'][0] # For eval (Metric)
            })

        # Create HuggingFace Dataset
        full_ds = Dataset.from_list(formatted_data)

        # Split strictly
        # Note: Some tasks are small, so we ensure we don't error out
        actual_train_size = min(sample_size, int(len(full_ds) * 0.8))
        actual_eval_size = min(eval_size, len(full_ds) - actual_train_size)

        split_ds = full_ds.train_test_split(train_size=actual_train_size, test_size=actual_eval_size, seed=42)
        print(f"  Final Split -> Train: {len(split_ds['train'])}, Eval: {len(split_ds['test'])}")
        return split_ds

# ==========================================
# 3. MODEL SETUP
# ==========================================
def load_base_model():
    print(f"\nLoading Base FalconMamba: {MODEL_ID}")

    # Mamba requires trust_remote_code=True
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache=False # Disable cache for training (gradient checkpointing compatibility)
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Right padding for training

    return model, tokenizer

def setup_lora(model):
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        # Mamba "Translation" of Query/Value targeting:
        # We target the input projections and the SSM state projections
        target_modules=["in_proj", "x_proj", "dt_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return get_peft_model(model, peft_config)

# ==========================================
# 4. TRAINING & EVAL LOOP (PHASE 2)
# ==========================================

def train_task(model, tokenizer, dataset, task_name, output_dir):
    print(f"\n>>> Training on {task_name} <<<")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=SEQ_LENGTH)

    tokenized_train = dataset["train"].map(tokenize_function, batched=True)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=LR,
        num_train_epochs=EPOCHS_PER_TASK,
        logging_steps=20,
        bf16=True, # A100 supports this native
        save_strategy="no", # We save manually at end of task
        report_to="none",
        gradient_checkpointing=True, # Save VRAM
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    return model

def evaluate_forgetting(model, tokenizer, eval_datasets):
    """
    Evaluates the current model on ALL tasks loaded so far to check for forgetting.
    """
    print("\n--- Evaluating Forgetting ---")
    model.eval()
    tokenizer.padding_side = "left" # Left padding for generation

    results = {}

    for t_name, ds in eval_datasets.items():
        print(f"Testing on {t_name} (Size: {len(ds['test'])}) ...")

        # Sample small batch for speed in this demo
        subset = ds['test'].select(range(20))
        inputs = tokenizer(subset["input_prompt"], return_tensors="pt", padding=True, truncation=True, max_length=SEQ_LENGTH).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode
        generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Simple exact match or "contains" check for now (ROUGE is better for final report)
        # We strip the input prompt from the generation to see the answer
        clean_preds = [gen.replace(prompt, "").strip() for gen, prompt in zip(generated, subset["input_prompt"])]

        # Just printing first 2 for sanity check
        print(f"  Ex 1 Ground Truth: {subset['ground_truth'][0]}")
        print(f"  Ex 1 Prediction:   {clean_preds[0][:100]}...")

        results[t_name] = "Done (Check Logs)"

    tokenizer.padding_side = "right" # Reset for training
    model.train()
    return results

# ==========================================
# 5. MAIN EXECUTION (CORRECTED)
# ==========================================

# A. Load Model
base_model, tokenizer = load_base_model()

# --- CRITICAL FIX FOR GRADIENT CHECKPOINTING ---
# This enables gradients on the input embeddings, ensuring the calculation
# graph isn't broken during the backward pass.
base_model.enable_input_require_grads()
# -----------------------------------------------

model = setup_lora(base_model)
model.print_trainable_parameters()

# B. Load Data
loader = SuperNILoader(tokenizer)
all_datasets = {}
for name, url in TASK_URLS.items():
    all_datasets[name] = loader.download_and_parse(name, url)

# C. The Continual Learning Loop
adapter_path = f"{OUTPUT_DIR}/adapters"

# Ensure output directory exists
os.makedirs(adapter_path, exist_ok=True)

for i, (task_name, dataset) in enumerate(all_datasets.items()):
    print(f"\n\n{'='*40}")
    print(f"STAGE {i+1}: {task_name}")
    print(f"{'='*40}")

    # 1. Train on current task
    model = train_task(model, tokenizer, dataset, task_name, f"{OUTPUT_DIR}/{task_name}")

    # 2. Evaluate on ALL tasks (to see if Task 1 drops after Task 2)
    evaluate_forgetting(model, tokenizer, all_datasets)

    # 3. Save Adapter Checkpoint
    save_path = f"{adapter_path}/after_{task_name}"
    model.save_pretrained(save_path)
    print(f"Adapter saved to {save_path}")

print("\n\nBASELINE EXPERIMENT COMPLETE.")
print("Analyze the logs above: Did the 'Ex 1 Prediction' for Task1_QA change/degrade after training on Task2 and Task3?")