import torch
from transformers import AutoTokenizer
from model import Transformer


class Generator():
    def __init__(self, max_new_tokens=50, checkpoint_path="model.pth"):
        print("prepping model")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        self.d_model = 128
        self.max_seq_len = 64
        self.vocab_size = len(self.tokenizer)

        self.model = Transformer(vocab_size=self.vocab_size, d_model=self.d_model, max_seq_len=self.max_seq_len)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt, max_new_tokens):
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            print(f"\n User: {prompt}")
            print("\n Model: ")
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    context = input_ids[:, -self.max_seq_len:] #sliding window
                    logits = self.model(context)

                    next_token_logits = logits[:,-1,:]

                    next_token = torch.argmax(next_token_logits, dim=1, keepdim=True)
                    #top p comming soon


                    input_ids = torch.cat((input_ids, next_token), dim=1)

            generated_text = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
            print(generated_text)
            print("-"*40)

if __name__ == "__main__":
    g = Generator()
    g.generate(prompt="Once upon a time", max_new_tokens=50)
    
