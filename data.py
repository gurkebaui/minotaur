import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class CustomDataset(Dataset):
    def __init__(self, numSamples=1000, seq_len=64):
        print("lade datenset")
        super().__init__()
        dataset = load_dataset("roneneland/TinyStories", split="train", streaming=True)

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        tokens_list = []
        count = 0

        for item in dataset:
            text = item["text"]
            encoded = self.tokenizer.encode(text)
            tokens_list.extend(encoded)
            count += 1
            if count >= max_samples:
                break

        all_tokens = torch.tensor(tokens_list, dtype=torch.long)

        chunk_size = seq_len +1
        num_chunks = len(all_tokens) // chunk_size

        all_tokens = all_tokens[:num_chunks * chunk_size]
        self.chunks = all_tokens.view(num_chunks, chunk_size)
        print(f"Chunks: {num_chunks} deren Länge{seq_len}")
        

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk[:-1], chunk[1:]
    
        
        
    
        
