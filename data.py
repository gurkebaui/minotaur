import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, numSamples):
        super().__init__()
        self.x_data = torch.randn(num_samples, 128)
        self.y_data = torch.randn(0,128, (num_samples,))

    def __len__(self):
        return len(self.x_data)
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
    
        
