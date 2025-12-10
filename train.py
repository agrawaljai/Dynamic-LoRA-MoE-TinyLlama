# # # train.py (FINAL, VERIFIED, AND FAIR EXPERIMENT)

# # import torch
# # from datasets import load_dataset, concatenate_datasets, DatasetDict
# # from peft import LoraConfig, get_peft_model
# # from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
# # from trl import SFTTrainer
# # import os

# # # --- CONFIGURATION ---
# # MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# # OUTPUT_DIR_CODE = "./lora_adapters/lora_expert_code"
# # OUTPUT_DIR_POETRY = "./lora_adapters/lora_expert_poetry"
# # OUTPUT_DIR_BASELINE = "./lora_adapters/lora_baseline"

# # def get_model_and_tokenizer():
# #     """Loads the 4-bit quantized model and tokenizer."""
# #     quantization_config = BitsAndBytesConfig(
# #         load_in_4bit=True,
# #         bnb_4bit_compute_dtype=torch.bfloat16,
# #         bnb_4bit_quant_type="nf4"
# #     )
# #     model = AutoModelForCausalLM.from_pretrained(
# #         MODEL_ID,
# #         quantization_config=quantization_config,
# #         device_map="auto",
# #         trust_remote_code=True,
# #     )
# #     model.config.use_cache = False
# #     tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# #     tokenizer.pad_token = tokenizer.eos_token
# #     tokenizer.padding_side = "right"
# #     return model, tokenizer

# # def create_prompt(sample, type):
# #     """Creates a formatted prompt for the model based on data type."""
# #     if type == "code":
# #         # CodeAlpaca uses 'prompt' and 'completion'
# #         return f"### Instruction:\n{sample['prompt']}\n\n### Response:\n{sample['completion']}"
# #     elif type == "poetry":
# #         # poem_sentiment uses 'verse_text'
# #         return f"### Instruction:\nWrite a poem about the following topic: {sample['verse_text']}\n\n### Response:\n{sample['verse_text']}"

# # def train_expert(model, tokenizer, dataset, output_dir, num_epochs=3):
# #     """Trains a LoRA expert on a given dataset with 4GB VRAM optimizations."""
# #     lora_config = LoraConfig(
# #         r=16,
# #         lora_alpha=32,
# #         target_modules=["q_proj", "v_proj"],
# #         lora_dropout=0.05,
# #         bias="none",
# #         task_type="CAUSAL_LM"
# #     )
# #     peft_model = get_peft_model(model, lora_config)
# #     peft_model.print_trainable_parameters()
    
# #     # --- 4GB VRAM OPTIMIZATIONS ---
# #     training_args = TrainingArguments(
# #         output_dir=output_dir,
# #         per_device_train_batch_size=1,
# #         gradient_accumulation_steps=16,
# #         learning_rate=2e-4,
# #         num_train_epochs=num_epochs, # Use the specified number of epochs
# #         logging_steps=10,
# #         save_total_limit=1,
# #         fp16=True,
# #     )
# #     trainer = SFTTrainer(
# #         model=peft_model,
# #         tokenizer=tokenizer,
# #         train_dataset=dataset,
# #         dataset_text_field="text",
# #         max_seq_length=512,
# #         args=training_args,
# #         packing=True,
# #     )
# #     print(f"--- Starting training for {output_dir} ({num_epochs} epochs) ---")
# #     trainer.train()
# #     print(f"--- Finished training. Saving model to {output_dir} ---")
# #     trainer.save_model(output_dir)

# # def load_and_prepare_datasets():
# #     """Loads and splits the datasets for a fair experiment."""
# #     print("Loading and splitting CodeAlpaca_20K...")
# #     # Use 90% for training, 10% for testing
# #     code_splits = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train").train_test_split(test_size=0.1, seed=42)
    
# #     print("Loading and splitting poem_sentiment...")
# #     # Use the full 'train' and 'test' splits
# #     poetry_train = load_dataset("poem_sentiment", split="train")
# #     poetry_test = load_dataset("poem_sentiment", split="test")
    
# #     # Manually create a DatasetDict for poetry
# #     poetry_splits = DatasetDict({
# #         'train': poetry_train,
# #         'test': poetry_test
# #     })

# #     return code_splits, poetry_splits

# # def main():
# #     # Load and split all data first
# #     code_splits, poetry_splits = load_and_prepare_datasets()

# #     # --- 1. Train Code Expert (on 90% of Code data for 3 Epochs) ---
# #     model, tokenizer = get_model_and_tokenizer()
# #     print("\n--- Preparing Code Expert ---")
# #     code_expert_train = code_splits['train'].map(lambda p: {"text": create_prompt(p, "code")}, remove_columns=list(code_splits['train'].features))
# #     print(f"Code Expert will train on {len(code_expert_train)} samples for 3 epochs.")
# #     train_expert(model, tokenizer, code_expert_train, OUTPUT_DIR_CODE, num_epochs=3)

# #     # --- 2. Train Poetry Expert (on 100% of Poetry data for 30 Epochs) ---
# #     model, tokenizer = get_model_and_tokenizer()
# #     print("\n--- Preparing Poetry Expert ---")
# #     poetry_expert_train = poetry_splits['train'].map(lambda p: {"text": create_prompt(p, "poetry")}, remove_columns=list(poetry_splits['train'].features))
# #     print(f"Poetry Expert will train on {len(poetry_expert_train)} samples for 30 EPOCHS.")
# #     # We over-train the poetry expert to make it a specialist
# #     train_expert(model, tokenizer, poetry_expert_train, OUTPUT_DIR_POETRY, num_epochs=30)

# #     # --- 3. Train Baseline Model (on 50% of Code + 50% of Poetry data, 3 Epochs) ---
# #     model, tokenizer = get_model_and_tokenizer()
# #     print("\n--- Preparing Baseline Model ---")
    
# #     # We take the *training* splits and halve them
# #     baseline_code_train = code_splits['train'].select(range(int(len(code_splits['train']) * 0.5)))
# #     baseline_poetry_train = poetry_splits['train'].select(range(int(len(poetry_splits['train']) * 0.5)))
    
# #     baseline_code_train = baseline_code_train.map(lambda p: {"text": create_prompt(p, "code")}, remove_columns=list(baseline_code_train.features))
# #     baseline_poetry_train = baseline_poetry_train.map(lambda p: {"text": create_prompt(p, "poetry")}, remove_columns=list(baseline_poetry_train.features))

# #     mixed_dataset = concatenate_datasets([baseline_code_train, baseline_poetry_train])
# #     mixed_dataset = mixed_dataset.shuffle(seed=42)
    
# #     print(f"Baseline Model will train on {len(mixed_dataset)} total samples for 3 epochs.")
# #     train_expert(model, tokenizer, mixed_dataset, OUTPUT_DIR_BASELINE, num_epochs=3)
    
# #     print("\n--- All training complete! ---")

# # if __name__ == "__main__":
# #     main()

# # train.py (MODIFIED to SKIP Code Expert, and FIX Poetry)

# import torch
# from datasets import load_dataset, concatenate_datasets, DatasetDict
# from peft import LoraConfig, get_peft_model
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
# from trl import SFTTrainer
# import os

# # --- CONFIGURATION ---
# MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# OUTPUT_DIR_CODE = "./lora_adapters/lora_expert_code"
# OUTPUT_DIR_POETRY = "./lora_adapters/lora_expert_poetry"
# OUTPUT_DIR_BASELINE = "./lora_adapters/lora_baseline"

# def get_model_and_tokenizer():
#     """Loads the 4-bit quantized model and tokenizer."""
#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_quant_type="nf4"
#     )
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_ID,
#         quantization_config=quantization_config,
#         device_map="auto",
#         trust_remote_code=True,
#     )
#     model.config.use_cache = False
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "right"
#     return model, tokenizer

# def create_prompt(sample, type):
#     """Creates a formatted prompt for the model based on data type."""
#     if type == "code":
#         # CodeAlpaca uses 'prompt' and 'completion'
#         return f"### Instruction:\n{sample['prompt']}\n\n### Response:\n{sample['completion']}"
#     elif type == "poetry":
#         # --- THIS IS THE FIX ---
#         # B-Rent-Data/Poetry_Instruction uses 'Instruction' and 'Response'
#         return f"### Instruction:\n{sample['Instruction']}\n\n### Response:\n{sample['Response']}"

# def train_expert(model, tokenizer, dataset, output_dir, num_epochs=3):
#     """Trains a LoRA expert on a given dataset with 4GB VRAM optimizations."""
#     lora_config = LoraConfig(
#         r=16,
#         lora_alpha=32,
#         target_modules=["q_proj", "v_proj"],
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM"
#     )
#     peft_model = get_peft_model(model, lora_config)
#     peft_model.print_trainable_parameters()
    
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=16,
#         learning_rate=2e-4,
#         num_train_epochs=num_epochs, # Use the specified number of epochs
#         logging_steps=10,
#         save_total_limit=1,
#         fp16=True,
#     )
#     trainer = SFTTrainer(
#         model=peft_model,
#         tokenizer=tokenizer,
#         train_dataset=dataset,
#         dataset_text_field="text",
#         max_seq_length=512,
#         args=training_args,
#         packing=True,
#     )
#     print(f"--- Starting training for {output_dir} ({num_epochs} epochs) ---")
#     trainer.train()
#     print(f"--- Finished training. Saving model to {output_dir} ---")
#     trainer.save_model(output_dir)

# def load_and_prepare_datasets():
#     """Loads and splits the datasets for a fair experiment."""
#     print("Loading and splitting CodeAlpaca_20K...")
#     code_splits = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train").train_test_split(test_size=0.1, seed=42)
    
#     # --- THIS IS THE FIX ---
#     print("Loading and splitting B-Rent-Data/Poetry_Instruction...")
#     poetry_splits = load_dataset("B-Rent-Data/Poetry_Instruction", split="train").train_test_split(test_size=0.1, seed=42)

#     return code_splits, poetry_splits

# def main():
#     # Load and split all data first
#     code_splits, poetry_splits = load_and_prepare_datasets()

#     # --- 1. TRAIN CODE EXPERT (SKIPPED!) ---
#     print("\n--- SKIPPING Code Expert Training (assuming already trained) ---")
#     if not os.path.exists(OUTPUT_DIR_CODE):
#         print(f"WARNING: Code expert not found at {OUTPUT_DIR_CODE}. Evaluation will fail.")
#     else:
#         print(f"Found existing code expert at {OUTPUT_DIR_CODE}.")


#     # --- 2. Train Poetry Expert (on 90% of REAL Poetry data for 15 Epochs) ---
#     model, tokenizer = get_model_and_tokenizer()
#     print("\n--- Preparing NEW Poetry Expert ---")
#     poetry_expert_train = poetry_splits['train'].map(lambda p: {"text": create_prompt(p, "poetry")}, remove_columns=list(poetry_splits['train'].features))
#     print(f"Poetry Expert will train on {len(poetry_expert_train)} samples for 15 EPOCHS.")
#     # We over-train the poetry expert to make it a specialist
#     train_expert(model, tokenizer, poetry_expert_train, OUTPUT_DIR_POETRY, num_epochs=15)

#     # --- 3. Train Baseline Model (on 50% of Code + 50% of REAL Poetry data, 3 Epochs) ---
#     model, tokenizer = get_model_and_tokenizer()
#     print("\n--- Preparing NEW Baseline Model ---")
    
#     # We take the *training* splits and halve them
#     baseline_code_train = code_splits['train'].select(range(int(len(code_splits['train']) * 0.5)))
#     baseline_poetry_train = poetry_splits['train'].select(range(int(len(poetry_splits['train']) * 0.5)))
    
#     baseline_code_train = baseline_code_train.map(lambda p: {"text": create_prompt(p, "code")}, remove_columns=list(baseline_code_train.features))
#     baseline_poetry_train = baseline_poetry_train.map(lambda p: {"text": create_prompt(p, "poetry")}, remove_columns=list(baseline_poetry_train.features))

#     mixed_dataset = concatenate_datasets([baseline_code_train, baseline_poetry_train])
#     mixed_dataset = mixed_dataset.shuffle(seed=42)
    
#     print(f"Baseline Model will train on {len(mixed_dataset)} total samples for 3 epochs.")
#     train_expert(model, tokenizer, mixed_dataset, OUTPUT_DIR_BASELINE, num_epochs=3)
    
#     print("\n--- All training complete! ---")

# if __name__ == "__main__":
#     main()

import torch
from datasets import load_dataset, concatenate_datasets, DatasetDict
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import os

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR_CODE = "./lora_adapters/lora_expert_code"
OUTPUT_DIR_POETRY = "./lora_adapters/lora_expert_poetry"
OUTPUT_DIR_BASELINE = "./lora_adapters/lora_baseline"

def get_model_and_tokenizer():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer

def create_prompt(sample, type):
    if type == "code":
        return f"### Instruction:\n{sample['prompt']}\n\n### Response:\n{sample['completion']}"
    elif type == "poetry":
        return f"### Instruction:\nWrite a poem about the following topic: {sample['verse_text']}\n\n### Response:\n{sample['verse_text']}"

def train_expert(model, tokenizer, dataset, output_dir, num_epochs):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    peft_model = get_peft_model(model, lora_config)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=num_epochs,
        logging_steps=10,
        save_total_limit=1,
        fp16=True,
    )
    trainer = SFTTrainer(
        model=peft_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        args=training_args,
        packing=True,
    )
    trainer.train()
    trainer.save_model(output_dir)

def load_and_prepare_datasets():
    code_splits = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train").train_test_split(test_size=0.1, seed=42)
    poetry_train = load_dataset("poem_sentiment", split="train")
    poetry_test = load_dataset("poem_sentiment", split="test")
    
    poetry_splits = DatasetDict({
        'train': poetry_train,
        'test': poetry_test
    })
    return code_splits, poetry_splits

def main():
    code_splits, poetry_splits = load_and_prepare_datasets()

    model, tokenizer = get_model_and_tokenizer()
    code_expert_train = code_splits['train'].map(lambda p: {"text": create_prompt(p, "code")}, remove_columns=list(code_splits['train'].features))
    train_expert(model, tokenizer, code_expert_train, OUTPUT_DIR_CODE, num_epochs=3)
    
    del model, tokenizer
    torch.cuda.empty_cache()

    model, tokenizer = get_model_and_tokenizer()
    poetry_expert_train = poetry_splits['train'].map(lambda p: {"text": create_prompt(p, "poetry")}, remove_columns=list(poetry_splits['train'].features))
    train_expert(model, tokenizer, poetry_expert_train, OUTPUT_DIR_POETRY, num_epochs=30)

    del model, tokenizer
    torch.cuda.empty_cache()

    model, tokenizer = get_model_and_tokenizer()
    baseline_code_train = code_splits['train'].select(range(int(len(code_splits['train']) * 0.5)))
    baseline_poetry_train = poetry_splits['train'].select(range(int(len(poetry_splits['train']) * 0.5)))
    
    baseline_code_train = baseline_code_train.map(lambda p: {"text": create_prompt(p, "code")}, remove_columns=list(baseline_code_train.features))
    baseline_poetry_train = baseline_poetry_train.map(lambda p: {"text": create_prompt(p, "poetry")}, remove_columns=list(baseline_poetry_train.features))

    mixed_dataset = concatenate_datasets([baseline_code_train, baseline_poetry_train])
    mixed_dataset = mixed_dataset.shuffle(seed=42)
    
    train_expert(model, tokenizer, mixed_dataset, OUTPUT_DIR_BASELINE, num_epochs=3)

if __name__ == "__main__":
    main()