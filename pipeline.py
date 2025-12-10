# # pipeline.py (Corrected - Removed invalid flag)

# import torch
# from peft import PeftModel
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from router import LoRARouter
# import os

# ADAPTER_CODE_PATH = "./lora_adapters/lora_expert_code"
# ADAPTER_POETRY_PATH = "./lora_adapters/lora_expert_poetry"

# class MoE_Pipeline:
#     def __init__(self):
#         self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.router = LoRARouter(device=self.device)

#         self.expert_map = {
#             "lora_expert_code": ADAPTER_CODE_PATH,
#             "lora_expert_poetry": ADAPTER_POETRY_PATH
#         }
        
#         for path in self.expert_map.values():
#             if not os.path.exists(path):
#                 raise FileNotFoundError(f"Adapter not found at {path}. Please run train.py first.")

#         self._load_model_and_adapters()

#     def _load_model_and_adapters(self):
#         print("Pipeline: Loading Base Model and Adapters...")
#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.bfloat16,
#             bnb_4bit_quant_type="nf4"
#         )
#         self.base_model = AutoModelForCausalLM.from_pretrained(
#             self.model_id,
#             quantization_config=quantization_config,
#             device_map="auto",
#             trust_remote_code=True
#             # llm_int8_enable_fp32_cpu_offload=True  <-- THIS LINE HAS BEEN REMOVED
#         )
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
#         self.tokenizer.pad_token = self.tokenizer.eos_token

#         self.model = PeftModel.from_pretrained(
#             self.base_model, 
#             self.expert_map["lora_expert_code"], 
#             adapter_name="lora_expert_code"
#         )
#         self.model.load_adapter(
#             self.expert_map["lora_expert_poetry"], 
#             adapter_name="lora_expert_poetry"
#         )

#         print("Pipeline: All models loaded successfully.")

#     def generate(self, prompt: str, max_new_tokens: int = 150):
#         formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
#         expert_name = self.router.route(prompt)
        
#         self.model.set_adapter(expert_name)

#         input_ids = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True).input_ids.to(self.device)
        
#         with torch.no_grad():
#             outputs = self.model.generate(
#                 input_ids=input_ids,
#                 max_new_tokens=max_new_tokens,
#                 do_sample=True,
#                 temperature=0.7,
#                 top_k=50,
#                 top_p=0.95,
#             )
        
#         response = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
#         response_clean = response.split("### Response:\n")[-1].strip()
        
#         return response_clean, expert_name

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from router import LoRARouter
import os

ADAPTER_CODE_PATH = "./lora_adapters/lora_expert_code"
ADAPTER_POETRY_PATH = "./lora_adapters/lora_expert_poetry"

class MoE_Pipeline:
    def __init__(self):
        self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.router = LoRARouter(device=self.device)

        self.expert_map = {
            "lora_expert_code": ADAPTER_CODE_PATH,
            "lora_expert_poetry": ADAPTER_POETRY_PATH
        }
        
        self._load_model_and_adapters()

    def _load_model_and_adapters(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = PeftModel.from_pretrained(
            self.base_model, 
            self.expert_map["lora_expert_code"], 
            adapter_name="lora_expert_code"
        )
        self.model.load_adapter(
            self.expert_map["lora_expert_poetry"], 
            adapter_name="lora_expert_poetry"
        )

    def generate(self, prompt: str, max_new_tokens: int = 150):
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        expert_name = self.router.route(prompt)
        
        self.model.set_adapter(expert_name)

        input_ids = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True).input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            )
        
        response = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
        response_clean = response.split("### Response:\n")[-1].strip()
        
        return response_clean, expert_name