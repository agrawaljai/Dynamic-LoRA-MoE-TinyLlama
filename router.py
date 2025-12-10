# router.py

from sentence_transformers import SentenceTransformer, util
import torch

class LoRARouter:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.expert_descriptions = {
            "lora_expert_code": "This expert is for generating and explaining computer programming code, algorithms, and data structures.",
            "lora_expert_poetry": "This expert is for writing poems, song lyrics, short stories, and creative, artistic, or emotional text."
        }
        self.experts = list(self.expert_descriptions.keys())
        print("Router: Encoding expert descriptions...")
        self.expert_embeddings = self.model.encode(
            [self.expert_descriptions[key] for key in self.experts],
            convert_to_tensor=True,
            device=self.device
        )

    def route(self, prompt: str) -> str:
        prompt_embedding = self.model.encode(prompt, convert_to_tensor=True, device=self.device)
        similarities = util.pytorch_cos_sim(prompt_embedding, self.expert_embeddings)[0]
        best_expert_index = torch.argmax(similarities)
        return self.experts[best_expert_index]