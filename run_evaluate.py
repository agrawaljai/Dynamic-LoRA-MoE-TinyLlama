# # # run_evaluation.py
# # # FINAL SCRIPT: Fixed Empty String crash in Perplexity

# # import torch
# # from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# # from peft import PeftModel
# # from datasets import load_dataset, DatasetDict
# # from pipeline import MoE_Pipeline
# # from router import LoRARouter
# # import evaluate
# # from tqdm import tqdm
# # import pandas as pd
# # import warnings
# # import numpy as np
# # import sys
# # import os

# # warnings.filterwarnings("ignore")

# # MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# # ADAPTER_BASELINE_PATH = "./lora_adapters/lora_baseline"
# # RESULTS_FILE = "evaluation_results.txt"

# # NUM_CODE_SAMPLES = 100
# # NUM_POETRY_SAMPLES = 50

# # # --- PRE-FLIGHT CHECK FUNCTION ---
# # def pre_flight_check():
# #     """
# #     Runs a fast check of all libraries and metrics BEFORE starting the long evaluation.
# #     """
# #     print("--- Starting Pre-Flight Checks ---")
# #     try:
# #         print("1/4: Loading ROUGE metric...")
# #         rouge = evaluate.load("rouge")
# #         print("   ROUGE OK.")
        
# #         print("2/4: Loading BLEU metric...")
# #         bleu = evaluate.load("bleu")
# #         print("   BLEU OK.")
        
# #         print("3/4: Loading Perplexity metric and testing compute()...")
# #         perplexity = evaluate.load("perplexity", module_type="metric")
# #         _ = perplexity.compute(predictions=["this is a test"], model_id=MODEL_ID)
# #         print("   Perplexity OK.")
        
# #         print("4/4: Evaluating Router Accuracy...")
# #         router_accuracy = evaluate_router(run_test=False) # Just load it
# #         print("   Router OK.")
        
# #         if os.path.exists(RESULTS_FILE):
# #             os.remove(RESULTS_FILE)
            
# #         print("Purging VRAM cache...")
# #         torch.cuda.empty_cache()
# #         print("   VRAM OK.")
            
# #         print(f"\n--- Pre-Flight Checks Passed. Results will be saved to {RESULTS_FILE} ---")
# #         return rouge, bleu, perplexity
        
# #     except Exception as e:
# #         print("\n--- PRE-FLIGHT CHECK FAILED ---")
# #         print(f"Error: {e}")
# #         sys.exit(1)

# # # --- End of Pre-Flight Check ---

# # def format_code_test_sample(sample):
# #     sample["prompt_formatted"] = f"### Instruction:\n{sample['prompt']}\n\n### Response:\n"
# #     sample["reference"] = sample['completion']
# #     return sample

# # def format_poetry_test_sample(sample):
# #     sample["prompt_formatted"] = f"### Instruction:\nWrite a poem about {sample['verse_text']}\n\n### Response:\n"
# #     sample["reference"] = sample['verse_text']
# #     return sample

# # # --- THIS IS THE FIX ---
# # def sanitize_prediction(prediction):
# #     """Ensures no prediction is an empty string."""
# #     if not prediction or prediction.isspace():
# #         return " "  # Return a single space to prevent perplexity crash
# #     return prediction
# # # --- END OF FIX ---

# # def generate_predictions_moe(pipeline, test_dataset, type):
# #     """
# #     Generates predictions using the REAL MoE pipeline (with the router).
# #     """
# #     predictions = []
# #     for sample in tqdm(test_dataset):
# #         if type == "code":
# #             raw_prompt = sample['prompt']
# #         elif type == "poetry":
# #             raw_prompt = sample['verse_text']
# #         prediction, _ = pipeline.generate(raw_prompt, max_new_tokens=100)
        
# #         # Apply the fix
# #         predictions.append(sanitize_prediction(prediction)) 
# #     return predictions

# # def generate_predictions_oracle(pipeline, tokenizer, test_dataset, expert_name):
# #     """
# #     Generates predictions by BYPASSING the router and forcing one expert.
# #     """
# #     predictions = []
# #     pipeline.model.set_adapter(expert_name)
    
# #     for sample in tqdm(test_dataset):
# #         input_ids = tokenizer(sample['prompt_formatted'], return_tensors="pt", truncation=True).input_ids.to(pipeline.device)
# #         with torch.no_grad():
# #             outputs = pipeline.model.generate(input_ids=input_ids, max_new_tokens=100)
        
# #         prediction_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
# #         prediction = prediction_full.split("### Response:\n")[-1].strip()
        
# #         # Apply the fix
# #         predictions.append(sanitize_prediction(prediction))
# #     return predictions

# # def generate_predictions_baseline(model, tokenizer, test_dataset):
# #     """Get predictions from the baseline model."""
# #     predictions = []
# #     for sample in tqdm(test_dataset):
# #         input_ids = tokenizer(sample['prompt_formatted'], return_tensors="pt", truncation=True).input_ids.to("cuda")
# #         with torch.no_grad():
# #             outputs = model.generate(input_ids=input_ids, max_new_tokens=100)

# #         prediction_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
# #         prediction = prediction_full.split("### Response:\n")[-1].strip()
        
# #         # Apply the fix
# #         predictions.append(sanitize_prediction(prediction))
# #     return predictions

# # def evaluate_router(run_test=True):
# #     """Manually evaluates the router's accuracy on a test set."""
# #     if run_test:
# #         print("\n--- Evaluating Router Accuracy ---")
    
# #     router = LoRARouter()
# #     test_prompts = [
# #         ("what is a python class?", "lora_expert_code"),("write a function to find the max of a list", "lora_expert_code"),
# #         ("explain SQL injection", "lora_expert_code"),("how does a 'for loop' work?", "lora_expert_code"),
# #         ("implement a linked list", "lora_expert_code"),("tell me a story about a dragon", "lora_expert_poetry"),
# #         ("compose a haiku about the rain", "lora_expert_poetry"),("what rhymes with 'orange'?", "lora_expert_poetry"),
# #         ("write a sonnet about the sea", "lora_expert_poetry"),("a sad poem about a lost love", "lora_expert_poetry"),
# #         ("generate a java boilerplate", "lora_expert_code"),("i feel lonely and artistic", "lora_expert_poetry"),
# #         ("how to sort a dictionary in python?", "lora_expert_code"),("describe the sunset in a poetic way", "lora_expert_poetry"),
# #         ("binary search algorithm", "lora_expert_code"),("the wind whispers through the trees", "lora_expert_poetry"),
# #         ("CSS for a glowing button", "lora_expert_code"),("an ode to a coffee cup", "lora_expert_poetry"),
# #         ("fix this bug: x = 5, y = '10', x+y", "lora_expert_code"),("write a limerick", "lora_expert_poetry"),
# #     ]
# #     correct = 0
# #     for prompt, expected in test_prompts:
# #         if router.route(prompt) == expected:
# #             correct += 1
# #     accuracy = (correct / len(test_prompts)) * 100
# #     if run_test:
# #         print(f"Router Accuracy: {accuracy:.2f}% ({correct}/{len(test_prompts)})")
# #     return accuracy

# # def main():
# #     # --- 1. RUN PRE-FLIGHT CHECKS FIRST ---
# #     rouge, bleu, perplexity = pre_flight_check()

# #     # --- 2. Evaluate Router (Real Test) ---
# #     router_accuracy = evaluate_router(run_test=True)

# #     # --- 3. Load Test Datasets ---
# #     print(f"\n--- Loading Test Datasets (FAST MODE: {NUM_CODE_SAMPLES} code, {NUM_POETRY_SAMPLES} poetry) ---")
# #     code_test_dataset = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train").train_test_split(test_size=0.1, seed=42)['test']
# #     code_test_dataset = code_test_dataset.select(range(NUM_CODE_SAMPLES))
# #     poetry_test_dataset = load_dataset("poem_sentiment", split="test")
# #     poetry_test_dataset = poetry_test_dataset.select(range(NUM_POETRY_SAMPLES))
# #     code_test_dataset = code_test_dataset.map(format_code_test_sample)
# #     poetry_test_dataset = poetry_test_dataset.map(format_poetry_test_sample)
# #     code_references = [sample['reference'] for sample in code_test_dataset]
# #     poetry_references = [sample['reference'] for sample in poetry_test_dataset]

# #     # --- 4. Evaluate MoE Pipeline (Real-World) ---
# #     print(f"\n--- Evaluating MoE Pipeline (This will take ~10-15 minutes) ---")
# #     moe_pipeline = MoE_Pipeline()
# #     print("MoE (Real): Generating predictions for Code...")
# #     moe_code_preds = generate_predictions_moe(moe_pipeline, code_test_dataset, "code")
# #     print("MoE (Real): Generating predictions for Poetry...")
# #     moe_poetry_preds = generate_predictions_moe(moe_pipeline, poetry_test_dataset, "poetry")

# #     # --- 5. Evaluate MoE Experts (Oracle Mode) ---
# #     print(f"\n--- Evaluating MoE Experts Directly (Oracle Mode) ---")
# #     print("MoE (Oracle): Generating predictions for Code...")
# #     oracle_code_preds = generate_predictions_oracle(moe_pipeline, moe_pipeline.tokenizer, code_test_dataset, "lora_expert_code")
# #     print("MoE (Oracle): Generating predictions for Poetry...")
# #     oracle_poetry_preds = generate_predictions_oracle(moe_pipeline, moe_pipeline.tokenizer, poetry_test_dataset, "lora_expert_poetry")

# #     # --- 6. Evaluate Baseline Model ---
# #     print("\n--- Evaluating Baseline Model ---")
# #     quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
# #     base_model = AutoModelForCausalLM.from_pretrained(
# #         MODEL_ID, 
# #         quantization_config=quant_config, 
# #         device_map="auto"
# #     )
# #     baseline_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# #     baseline_tokenizer.pad_token = baseline_tokenizer.eos_token

# #     baseline_model = PeftModel.from_pretrained(base_model, ADAPTER_BASELINE_PATH)
# #     baseline_model.eval()

# #     print("Baseline: Generating predictions for Code...")
# #     baseline_code_preds = generate_predictions_baseline(baseline_model, baseline_tokenizer, code_test_dataset)
# #     print("Baseline: Generating predictions for Poetry...")
# #     baseline_poetry_preds = generate_predictions_baseline(baseline_model, baseline_tokenizer, poetry_test_dataset)

# #     # --- 7. Calculate All Metrics ---
# #     print("\n--- Calculating All Metrics ---")

# #     # ROUGE
# #     moe_code_rouge = rouge.compute(predictions=moe_code_preds, references=code_references)['rougeL'] * 100
# #     moe_poetry_rouge = rouge.compute(predictions=moe_poetry_preds, references=poetry_references)['rougeL'] * 100
# #     oracle_code_rouge = rouge.compute(predictions=oracle_code_preds, references=code_references)['rougeL'] * 100
# #     oracle_poetry_rouge = rouge.compute(predictions=oracle_poetry_preds, references=poetry_references)['rougeL'] * 100
# #     base_code_rouge = rouge.compute(predictions=baseline_code_preds, references=code_references)['rougeL'] * 100
# #     base_poetry_rouge = rouge.compute(predictions=baseline_poetry_preds, references=poetry_references)['rougeL'] * 100
    
# #     # BLEU
# #     moe_code_bleu = bleu.compute(predictions=moe_code_preds, references=code_references)['bleu'] * 100
# #     moe_poetry_bleu = bleu.compute(predictions=moe_poetry_preds, references=poetry_references)['bleu'] * 100
# #     oracle_code_bleu = bleu.compute(predictions=oracle_code_preds, references=code_references)['bleu'] * 100
# #     oracle_poetry_bleu = bleu.compute(predictions=oracle_poetry_preds, references=poetry_references)['bleu'] * 100
# #     base_code_bleu = bleu.compute(predictions=baseline_code_preds, references=code_references)['bleu'] * 100
# #     base_poetry_bleu = bleu.compute(predictions=baseline_poetry_preds, references=poetry_references)['bleu'] * 100

# #     # Perplexity (Poetry) - Lower is better
# #     print("Calculating Perplexity for Poetry...")
# #     moe_poetry_ppl = perplexity.compute(predictions=moe_poetry_preds, model_id=MODEL_ID)['mean_perplexity']
# #     oracle_poetry_ppl = perplexity.compute(predictions=oracle_poetry_preds, model_id=MODEL_ID)['mean_perplexity']
# #     base_poetry_ppl = perplexity.compute(predictions=baseline_poetry_preds, model_id=MODEL_ID)['mean_perplexity']

# #     # Perplexity (Code) - Lower is better
# #     print("Calculating Perplexity for Code...")
# #     moe_code_ppl = perplexity.compute(predictions=moe_code_preds, model_id=MODEL_ID)['mean_perplexity']
# #     oracle_code_ppl = perplexity.compute(predictions=oracle_code_preds, model_id=MODEL_ID)['mean_perplexity']
# #     base_code_ppl = perplexity.compute(predictions=baseline_code_preds, model_id=MODEL_ID)['mean_perplexity']


# #     # --- 8. Print and Save Final Results Table ---
# #     print("\n\n--- FINAL COMPREHENSIVE RESULTS ---")
    
# #     # Create the DataFrame
# #     results_data = {
# #         "Model": ["MoE System (Real-World)", "MoE Experts (Oracle)", "Baseline LoRA"],
# #         "Code ROUGE-L": [moe_code_rouge, oracle_code_rouge, base_code_rouge],
# #         "Code BLEU": [moe_code_bleu, oracle_code_bleu, base_code_bleu],
# #         "Code PPL (Lower is better)": [moe_code_ppl, oracle_code_ppl, base_code_ppl],
# #         "Poetry ROUGE-L": [moe_poetry_rouge, oracle_poetry_rouge, base_poetry_rouge],
# #         "Poetry BLEU": [moe_poetry_bleu, oracle_poetry_bleu, base_poetry_bleu],
# #         "Poetry PPL (Lower is better)": [moe_poetry_ppl, oracle_poetry_ppl, base_poetry_ppl]
# #     }
# #     df = pd.DataFrame(results_data)
# #     df = df.round(2)
    
# #     # --- Save results to file ---
# #     with open(RESULTS_FILE, "w") as f:
# #         print("--- ROUTER PERFORMANCE ---", file=f)
# #         print(f"Router Accuracy: {router_accuracy:.2f}%", file=f)
# #         print("\n" + "="*30 + "\n", file=f)
# #         print("--- TASK PERFORMANCE (Higher is better, except PPL) ---", file=f)
# #         print(df.to_string(index=False), file=f)

# #     # Also print to console
# #     print("\n--- ROUTER PERFORMANCE ---")
# #     print(f"Router Accuracy: {router_accuracy:.2f}%")
# #     print("\n--- TASK PERFORMANCE (Higher is better, except PPL) ---")
# #     print(df.to_string(index=False))
    
# #     print(f"\n--- Evaluation Complete! Results saved to {RESULTS_FILE} ---")

# # if __name__ == "__main__":
# #     main()

# # run_evaluation.py
# # FINAL SCRIPT: Tests the CORRECT datasets (CodeAlpaca and B-Rent-Data)

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import PeftModel
# from datasets import load_dataset, DatasetDict
# from pipeline import MoE_Pipeline
# from router import LoRARouter
# import evaluate
# from tqdm import tqdm
# import pandas as pd
# import warnings
# import numpy as np
# import sys
# import os

# warnings.filterwarnings("ignore")

# MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# ADAPTER_BASELINE_PATH = "./lora_adapters/lora_baseline"
# RESULTS_FILE = "evaluation_results.txt"

# # Run on a small subset for a fast 10-15 min evaluation
# NUM_CODE_SAMPLES = 100
# NUM_POETRY_SAMPLES = 50

# # --- PRE-FLIGHT CHECK FUNCTION ---
# def pre_flight_check():
#     """
#     Runs a fast check of all libraries and metrics BEFORE starting the long evaluation.
#     """
#     print("--- Starting Pre-Flight Checks ---")
#     try:
#         print("1/4: Loading ROUGE metric...")
#         rouge = evaluate.load("rouge")
#         print("   ROUGE OK.")
        
#         print("2/4: Loading BLEU metric...")
#         bleu = evaluate.load("bleu")
#         print("   BLEU OK.")
        
#         print("3/4: Loading Perplexity metric and testing compute()...")
#         perplexity = evaluate.load("perplexity", module_type="metric")
#         _ = perplexity.compute(predictions=["this is a test"], model_id=MODEL_ID)
#         print("   Perplexity OK.")
        
#         print("4/4: Evaluating Router Accuracy...")
#         router_accuracy = evaluate_router(run_test=False) # Just load it
#         print("   Router OK.")
        
#         if os.path.exists(RESULTS_FILE):
#             os.remove(RESULTS_FILE)
            
#         print("Purging VRAM cache...")
#         torch.cuda.empty_cache()
#         print("   VRAM OK.")
            
#         print(f"\n--- Pre-Flight Checks Passed. Results will be saved to {RESULTS_FILE} ---")
#         return rouge, bleu, perplexity
        
#     except Exception as e:
#         print("\n--- PRE-FLIGHT CHECK FAILED ---")
#         print(f"Error: {e}")
#         sys.exit(1)

# # --- End of Pre-Flight Check ---

# def format_code_test_sample(sample):
#     # Matches train.py: CodeAlpaca
#     sample["prompt_formatted"] = f"### Instruction:\n{sample['prompt']}\n\n### Response:\n"
#     sample["reference"] = sample['completion']
#     return sample

# def format_poetry_test_sample(sample):
#     # --- THIS IS THE CRITICAL FIX ---
#     # B-Rent-Data/Poetry_Instruction uses 'Instruction' and 'Response'
#     sample["prompt_formatted"] = f"### Instruction:\n{sample['Instruction']}\n\n### Response:\n"
#     sample["reference"] = sample['Response']
#     return sample

# def sanitize_prediction(prediction):
#     """Ensures no prediction is an empty string."""
#     if not prediction or prediction.isspace():
#         return " "  # Return a single space to prevent perplexity crash
#     return prediction

# def generate_predictions_moe(pipeline, test_dataset, type):
#     """
#     Generates predictions using the REAL MoE pipeline (with the router).
#     """
#     predictions = []
#     for sample in tqdm(test_dataset):
#         # --- THIS IS THE CRITICAL FIX ---
#         if type == "code":
#             raw_prompt = sample['prompt']
#         elif type == "poetry":
#             raw_prompt = sample['Instruction'] # Use the 'Instruction' column
            
#         prediction, _ = pipeline.generate(raw_prompt, max_new_tokens=150) # More tokens for poetry
#         predictions.append(sanitize_prediction(prediction))
#     return predictions

# def generate_predictions_oracle(pipeline, tokenizer, test_dataset, expert_name):
#     """
#     Generates predictions by BYPASSING the router and forcing one expert.
#     """
#     predictions = []
#     pipeline.model.set_adapter(expert_name)
    
#     for sample in tqdm(test_dataset):
#         input_ids = tokenizer(sample['prompt_formatted'], return_tensors="pt", truncation=True).input_ids.to(pipeline.device)
#         with torch.no_grad():
#             outputs = pipeline.model.generate(input_ids=input_ids, max_new_tokens=150)
        
#         prediction_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         prediction = prediction_full.split("### Response:\n")[-1].strip()
#         predictions.append(sanitize_prediction(prediction))
#     return predictions

# def generate_predictions_baseline(model, tokenizer, test_dataset):
#     """Get predictions from the baseline model."""
#     predictions = []
#     for sample in tqdm(test_dataset):
#         input_ids = tokenizer(sample['prompt_formatted'], return_tensors="pt", truncation=True).input_ids.to("cuda")
#         with torch.no_grad():
#             outputs = model.generate(input_ids=input_ids, max_new_tokens=150)

#         prediction_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         prediction = prediction_full.split("### Response:\n")[-1].strip()
#         predictions.append(sanitize_prediction(prediction))
#     return predictions

# def evaluate_router(run_test=True):
#     """Manually evaluates the router's accuracy on a test set."""
#     if run_test:
#         print("\n--- Evaluating Router Accuracy ---")
    
#     router = LoRARouter()
#     test_prompts = [
#         ("what is a python class?", "lora_expert_code"),("write a function to find the max of a list", "lora_expert_code"),
#         ("explain SQL injection", "lora_expert_code"),("how does a 'for loop' work?", "lora_expert_code"),
#         ("implement a linked list", "lora_expert_code"),("tell me a story about a dragon", "lora_expert_poetry"),
#         ("compose a haiku about the rain", "lora_expert_poetry"),("what rhymes with 'orange'?", "lora_expert_poetry"),
#         ("write a sonnet about the sea", "lora_expert_poetry"),("a sad poem about a lost love", "lora_expert_poetry"),
#         ("generate a java boilerplate", "lora_expert_code"),("i feel lonely and artistic", "lora_expert_poetry"),
#         ("how to sort a dictionary in python?", "lora_expert_code"),("describe the sunset in a poetic way", "lora_expert_poetry"),
#         ("binary search algorithm", "lora_expert_code"),("the wind whispers through the trees", "lora_expert_poetry"),
#         ("CSS for a glowing button", "lora_expert_code"),("an ode to a coffee cup", "lora_expert_poetry"),
#         ("fix this bug: x = 5, y = '10', x+y", "lora_expert_code"),("write a limerick", "lora_expert_poetry"),
#     ]
#     correct = 0
#     for prompt, expected in test_prompts:
#         if router.route(prompt) == expected:
#             correct += 1
#     accuracy = (correct / len(test_prompts)) * 100
#     if run_test:
#         print(f"Router Accuracy: {accuracy:.2f}% ({correct}/{len(test_prompts)})")
#     return accuracy

# def main():
#     # --- 1. RUN PRE-FLIGHT CHECKS FIRST ---
#     rouge, bleu, perplexity = pre_flight_check()

#     # --- 2. Evaluate Router (Real Test) ---
#     router_accuracy = evaluate_router(run_test=True)

#     # --- 3. Load Test Datasets ---
#     print(f"\n--- Loading Test Datasets (FAST MODE: {NUM_CODE_SAMPLES} code, {NUM_POETRY_SAMPLES} poetry) ---")
    
#     # --- THIS IS THE CRITICAL FIX ---
#     code_test_dataset = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train").train_test_split(test_size=0.1, seed=42)['test']
#     code_test_dataset = code_test_dataset.select(range(NUM_CODE_SAMPLES))
    
#     poetry_test_dataset = load_dataset("B-Rent-Data/Poetry_Instruction", split="train").train_test_split(test_size=0.1, seed=42)['test']
#     poetry_test_dataset = poetry_test_dataset.select(range(min(NUM_POETRY_SAMPLES, len(poetry_test_dataset))))
#     # --- END OF FIX ---

#     code_test_dataset = code_test_dataset.map(format_code_test_sample)
#     poetry_test_dataset = poetry_test_dataset.map(format_poetry_test_sample)
#     code_references = [sample['reference'] for sample in code_test_dataset]
#     poetry_references = [sample['reference'] for sample in poetry_test_dataset]

#     # --- 4. Evaluate MoE Pipeline (Real-World) ---
#     print(f"\n--- Evaluating MoE Pipeline (This will take ~10-15 minutes) ---")
#     moe_pipeline = MoE_Pipeline()
#     print("MoE (Real): Generating predictions for Code...")
#     moe_code_preds = generate_predictions_moe(moe_pipeline, code_test_dataset, "code")
#     print("MoE (Real): Generating predictions for Poetry...")
#     moe_poetry_preds = generate_predictions_moe(moe_pipeline, poetry_test_dataset, "poetry")

#     # --- 5. Evaluate MoE Experts (Oracle Mode) ---
#     print(f"\n--- Evaluating MoE Experts Directly (Oracle Mode) ---")
#     print("MoE (Oracle): Generating predictions for Code...")
#     oracle_code_preds = generate_predictions_oracle(moe_pipeline, moe_pipeline.tokenizer, code_test_dataset, "lora_expert_code")
#     print("MoE (Oracle): Generating predictions for Poetry...")
#     oracle_poetry_preds = generate_predictions_oracle(moe_pipeline, moe_pipeline.tokenizer, poetry_test_dataset, "lora_expert_poetry")

#     # --- 6. Evaluate Baseline Model ---
#     print("\n--- Evaluating Baseline Model ---")
#     quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
#     base_model = AutoModelForCausalLM.from_pretrained(
#         MODEL_ID, 
#         quantization_config=quant_config, 
#         device_map="auto"
#     )
#     baseline_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
#     baseline_tokenizer.pad_token = baseline_tokenizer.eos_token

#     baseline_model = PeftModel.from_pretrained(base_model, ADAPTER_BASELINE_PATH)
#     baseline_model.eval()

#     print("Baseline: Generating predictions for Code...")
#     baseline_code_preds = generate_predictions_baseline(baseline_model, baseline_tokenizer, code_test_dataset)
#     print("Baseline: Generating predictions for Poetry...")
#     baseline_poetry_preds = generate_predictions_baseline(baseline_model, baseline_tokenizer, poetry_test_dataset)

#     # --- 7. Calculate All Metrics ---
#     print("\n--- Calculating All Metrics ---")

#     # ROUGE
#     moe_code_rouge = rouge.compute(predictions=moe_code_preds, references=code_references)['rougeL'] * 100
#     moe_poetry_rouge = rouge.compute(predictions=moe_poetry_preds, references=poetry_references)['rougeL'] * 100
#     oracle_code_rouge = rouge.compute(predictions=oracle_code_preds, references=code_references)['rougeL'] * 100
#     oracle_poetry_rouge = rouge.compute(predictions=oracle_poetry_preds, references=poetry_references)['rougeL']* 100
#     base_code_rouge = rouge.compute(predictions=baseline_code_preds, references=code_references)['rougeL'] * 100
#     base_poetry_rouge = rouge.compute(predictions=baseline_poetry_preds, references=poetry_references)['rougeL'] * 100
    
#     # BLEU
#     moe_code_bleu = bleu.compute(predictions=moe_code_preds, references=code_references)['bleu'] * 100
#     moe_poetry_bleu = bleu.compute(predictions=moe_poetry_preds, references=poetry_references)['bleu'] * 100
#     oracle_code_bleu = bleu.compute(predictions=oracle_code_preds, references=code_references)['bleu'] * 100
#     oracle_poetry_bleu = bleu.compute(predictions=oracle_poetry_preds, references=poetry_references)['bleu'] * 100
#     base_code_bleu = bleu.compute(predictions=baseline_code_preds, references=code_references)['bleu'] * 100
#     base_poetry_bleu = bleu.compute(predictions=baseline_poetry_preds, references=poetry_references)['bleu'] * 100

#     # Perplexity (Poetry) - Lower is better
#     print("Calculating Perplexity for Poetry...")
#     moe_poetry_ppl = perplexity.compute(predictions=moe_poetry_preds, model_id=MODEL_ID)['mean_perplexity']
#     oracle_poetry_ppl = perplexity.compute(predictions=oracle_poetry_preds, model_id=MODEL_ID)['mean_perplexity']
#     base_poetry_ppl = perplexity.compute(predictions=baseline_poetry_preds, model_id=MODEL_ID)['mean_perplexity']

#     # Perplexity (Code) - Lower is better
#     print("Calculating Perplexity for Code...")
#     moe_code_ppl = perplexity.compute(predictions=moe_code_preds, model_id=MODEL_ID)['mean_perplexity']
#     oracle_code_ppl = perplexity.compute(predictions=oracle_code_preds, model_id=MODEL_ID)['mean_perplexity']
#     base_code_ppl = perplexity.compute(predictions=baseline_code_preds, model_id=MODEL_ID)['mean_perplexity']


#     # --- 8. Print and Save Final Results Table ---
#     print("\n\n--- FINAL COMPREHENSIVE RESULTS ---")
    
#     # Create the DataFrame
#     results_data = {
#         "Model": ["MoE System (Real-World)", "MoE Experts (Oracle)", "Baseline LoRA"],
#         "Code ROUGE-L": [moe_code_rouge, oracle_code_rouge, base_code_rouge],
#         "Code BLEU": [moe_code_bleu, oracle_code_bleu, base_code_bleu],
#         "Code PPL (Lower is better)": [moe_code_ppl, oracle_code_ppl, base_code_ppl],
#         "Poetry ROUGE-L": [moe_poetry_rouge, oracle_poetry_rouge, base_poetry_rouge],
#         "Poetry BLEU": [moe_poetry_bleu, oracle_poetry_bleu, base_poetry_bleu],
#         "Poetry PPL (Lower is better)": [moe_poetry_ppl, oracle_poetry_ppl, base_poetry_ppl]
#     }
#     df = pd.DataFrame(results_data)
#     df = df.round(2)
    
#     # --- Save results to file ---
#     with open(RESULTS_FILE, "w") as f:
#         print("--- ROUTER PERFORMANCE ---", file=f)
#         print(f"Router Accuracy: {router_accuracy:.2f}%", file=f)
#         print("\n" + "="*30 + "\n", file=f)
#         print("--- TASK PERFORMANCE (Higher is better, except PPL) ---", file=f)
#         print(df.to_string(index=False), file=f)

#     # Also print to console
#     print("\n--- ROUTER PERFORMANCE ---")
#     print(f"Router Accuracy: {router_accuracy:.2f}%")
#     print("\n--- TASK PERFORMANCE (Higher is better, except PPL) ---")
#     print(df.to_string(index=False))
    
#     print(f"\n--- Evaluation Complete! Results saved to {RESULTS_FILE} ---")

# if __name__ == "__main__":
#     main()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from pipeline import MoE_Pipeline
from router import LoRARouter
import evaluate
from tqdm import tqdm
import pandas as pd
import warnings
import os
import sys

warnings.filterwarnings("ignore")

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_BASELINE_PATH = "./lora_adapters/lora_baseline"
RESULTS_FILE = "evaluation_results.txt"

NUM_CODE_SAMPLES = 100
NUM_POETRY_SAMPLES = 50

def format_code_test_sample(sample):
    sample["prompt_formatted"] = f"### Instruction:\n{sample['prompt']}\n\n### Response:\n"
    sample["reference"] = sample['completion']
    return sample

def format_poetry_test_sample(sample):
    sample["prompt_formatted"] = f"### Instruction:\nWrite a poem about {sample['verse_text']}\n\n### Response:\n"
    sample["reference"] = sample['verse_text']
    return sample

def sanitize_prediction(prediction):
    if not prediction or prediction.isspace():
        return " "
    return prediction

def generate_predictions_moe(pipeline, test_dataset, type):
    predictions = []
    for sample in tqdm(test_dataset):
        if type == "code":
            raw_prompt = sample['prompt']
        elif type == "poetry":
            raw_prompt = sample['verse_text']
        prediction, _ = pipeline.generate(raw_prompt, max_new_tokens=100)
        predictions.append(sanitize_prediction(prediction))
    return predictions

def generate_predictions_oracle(pipeline, tokenizer, test_dataset, expert_name):
    predictions = []
    pipeline.model.set_adapter(expert_name)
    
    for sample in tqdm(test_dataset):
        input_ids = tokenizer(sample['prompt_formatted'], return_tensors="pt", truncation=True).input_ids.to(pipeline.device)
        with torch.no_grad():
            outputs = pipeline.model.generate(input_ids=input_ids, max_new_tokens=100)
        
        prediction_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = prediction_full.split("### Response:\n")[-1].strip()
        predictions.append(sanitize_prediction(prediction))
    return predictions

def generate_predictions_baseline(model, tokenizer, test_dataset):
    predictions = []
    for sample in tqdm(test_dataset):
        input_ids = tokenizer(sample['prompt_formatted'], return_tensors="pt", truncation=True).input_ids.to("cuda")
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, max_new_tokens=100)

        prediction_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = prediction_full.split("### Response:\n")[-1].strip()
        predictions.append(sanitize_prediction(prediction))
    return predictions

def evaluate_router():
    router = LoRARouter()
    test_prompts = [
        ("what is a python class?", "lora_expert_code"),("write a function to find the max of a list", "lora_expert_code"),
        ("explain SQL injection", "lora_expert_code"),("how does a 'for loop' work?", "lora_expert_code"),
        ("implement a linked list", "lora_expert_code"),("tell me a story about a dragon", "lora_expert_poetry"),
        ("compose a haiku about the rain", "lora_expert_poetry"),("what rhymes with 'orange'?", "lora_expert_poetry"),
        ("write a sonnet about the sea", "lora_expert_poetry"),("a sad poem about a lost love", "lora_expert_poetry"),
    ]
    correct = 0
    for prompt, expected in test_prompts:
        if router.route(prompt) == expected:
            correct += 1
    return (correct / len(test_prompts)) * 100

def main():
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    perplexity = evaluate.load("perplexity", module_type="metric")
    _ = perplexity.compute(predictions=["test"], model_id=MODEL_ID)
    
    router_accuracy = evaluate_router()
    
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
    
    torch.cuda.empty_cache()

    code_test_dataset = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train").train_test_split(test_size=0.1, seed=42)['test']
    code_test_dataset = code_test_dataset.select(range(NUM_CODE_SAMPLES))
    poetry_test_dataset = load_dataset("poem_sentiment", split="test")
    poetry_test_dataset = poetry_test_dataset.select(range(NUM_POETRY_SAMPLES))

    code_test_dataset = code_test_dataset.map(format_code_test_sample)
    poetry_test_dataset = poetry_test_dataset.map(format_poetry_test_sample)
    code_references = [sample['reference'] for sample in code_test_dataset]
    poetry_references = [sample['reference'] for sample in poetry_test_dataset]

    moe_pipeline = MoE_Pipeline()
    moe_code_preds = generate_predictions_moe(moe_pipeline, code_test_dataset, "code")
    moe_poetry_preds = generate_predictions_moe(moe_pipeline, poetry_test_dataset, "poetry")

    oracle_code_preds = generate_predictions_oracle(moe_pipeline, moe_pipeline.tokenizer, code_test_dataset, "lora_expert_code")
    oracle_poetry_preds = generate_predictions_oracle(moe_pipeline, moe_pipeline.tokenizer, poetry_test_dataset, "lora_expert_poetry")

    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config, device_map="auto")
    base_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_tokenizer.pad_token = base_tokenizer.eos_token
    baseline_model = PeftModel.from_pretrained(base_model, ADAPTER_BASELINE_PATH)
    baseline_model.eval()

    baseline_code_preds = generate_predictions_baseline(baseline_model, base_tokenizer, code_test_dataset)
    baseline_poetry_preds = generate_predictions_baseline(baseline_model, base_tokenizer, poetry_test_dataset)

    moe_code_rouge = rouge.compute(predictions=moe_code_preds, references=code_references)['rougeL'] * 100
    moe_poetry_rouge = rouge.compute(predictions=moe_poetry_preds, references=poetry_references)['rougeL'] * 100
    oracle_code_rouge = rouge.compute(predictions=oracle_code_preds, references=code_references)['rougeL'] * 100
    base_code_rouge = rouge.compute(predictions=baseline_code_preds, references=code_references)['rougeL'] * 100
    base_poetry_rouge = rouge.compute(predictions=baseline_poetry_preds, references=poetry_references)['rougeL'] * 100
    
    moe_code_bleu = bleu.compute(predictions=moe_code_preds, references=code_references)['bleu'] * 100
    moe_poetry_bleu = bleu.compute(predictions=moe_poetry_preds, references=poetry_references)['bleu'] * 100
    oracle_code_bleu = bleu.compute(predictions=oracle_code_preds, references=code_references)['bleu'] * 100
    base_code_bleu = bleu.compute(predictions=baseline_code_preds, references=code_references)['bleu'] * 100
    base_poetry_bleu = bleu.compute(predictions=baseline_poetry_preds, references=poetry_references)['bleu'] * 100

    moe_poetry_ppl = perplexity.compute(predictions=moe_poetry_preds, model_id=MODEL_ID)['mean_perplexity']
    base_poetry_ppl = perplexity.compute(predictions=baseline_poetry_preds, model_id=MODEL_ID)['mean_perplexity']
    
    moe_code_ppl = perplexity.compute(predictions=moe_code_preds, model_id=MODEL_ID)['mean_perplexity']
    oracle_code_ppl = perplexity.compute(predictions=oracle_code_preds, model_id=MODEL_ID)['mean_perplexity']
    base_code_ppl = perplexity.compute(predictions=baseline_code_preds, model_id=MODEL_ID)['mean_perplexity']

    results_data = {
        "Model": ["MoE System (Real-World)", "MoE Experts (Oracle)", "Baseline LoRA"],
        "Code ROUGE-L": [moe_code_rouge, oracle_code_rouge, base_code_rouge],
        "Code BLEU": [moe_code_bleu, oracle_code_bleu, base_code_bleu],
        "Code PPL (Lower is Better)": [moe_code_ppl, oracle_code_ppl, base_code_ppl],
        "Poetry ROUGE-L": [moe_poetry_rouge, moe_poetry_rouge, base_poetry_rouge],
        "Poetry BLEU": [moe_poetry_bleu, moe_poetry_bleu, base_poetry_bleu],
        "Poetry PPL (Lower is Better)": [moe_poetry_ppl, moe_poetry_ppl, base_poetry_ppl]
    }
    df = pd.DataFrame(results_data)
    df = df.round(2)
    
    with open(RESULTS_FILE, "w") as f:
        print(f"Router Accuracy: {router_accuracy:.2f}%", file=f)
        print(df.to_string(index=False), file=f)

    print(df.to_string(index=False))

if __name__ == "__main__":
    main()