import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if not sys.path or sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)

import torch
from transformers import AutoTokenizer
from Transformer.model import Transformer
import torch.nn.functional as F
from config import ModelConfig




def sample_next_token(logits, temperature=0.8, top_k=50, top_p=0.9):
    logits = logits / temperature

    if(top_k > 0):
        top_k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, top_k)
        min_logit = v[:,[-1]]
        logits[logits < min_logit] = float('-inf')

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[...,1:] = sorted_indices_to_remove[...,:-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

class Generator():
    def __init__(self, cfg, max_new_tokens=None, checkpoint_path=None):
        print("prepping model")
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

        self.d_model = cfg.d_model
        self.max_seq_len = cfg.max_seq_len
        self.max_new_tokens = cfg.max_new_tokens if max_new_tokens is None else max_new_tokens
        self.vocab_size = len(self.tokenizer)

        ckpt = cfg.checkpoint_path if checkpoint_path is None else checkpoint_path
        self.model = Transformer(cfg)
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    
    def generate(self, prompt, max_new_tokens=None):
            max_new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            print(f"\n User: {prompt}")
            print("\n Model: ")
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    context = input_ids[:, -self.max_seq_len:] #sliding window
                    logits = self.model(context)

                    next_token_logits = logits[:,-1,:]

                    #next_token = torch.argmax(next_token_logits, dim=1, keepdim=True) # depreciated
                    next_token = sample_next_token(
                                                   next_token_logits,
                                                    temperature=self.cfg.temperature,
                                                     top_k=self.cfg.top_k,
                                                    top_p=self.cfg.top_p)

                    input_ids = torch.cat((input_ids, next_token), dim=1)

            generated_text = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
            print(generated_text)
            print("-"*40)


   
    

if __name__ == "__main__":
    cfg = ModelConfig()
    g = Generator(cfg)
    g.generate(prompt="Once upon a time", max_new_tokens=50)

    
