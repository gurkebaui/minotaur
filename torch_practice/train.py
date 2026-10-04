import model
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn


 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, num_samples):
        super().__init__()
        self.x_data = torch.randn(num_samples, 3072)
        self.y_data = torch.randint(0, 10, (num_samples,))

    def __len__(self):
        return len(self.x_data)
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

class Train:
    def __init__(self):
        super().__init__()
        self.model = model.ImageClassifier().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.dataset = CustomDataset(2000)
        self.dataloader = DataLoader(self.dataset, batch_size=64, shuffle=True)
        self.test_dataset = CustomDataset(500)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)
        
    def train(self):
        for epoch in range(10):
            for batch_x, batch_y in self.dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                self.optimizer.zero_grad()
                self.outputs = self.model(batch_x)
                self.loss = self.criterion(self.outputs, batch_y)
                self.loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1} - Loss: {self.loss.item():.4f}")

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.test_dataloader:

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = self.model(batch_x)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)

        accuracy = correct/total
        print(f"Test Accuracy: {accuracy * 100:.2f}")
        self.model.train()
        torch.save(self.model.state_dict(), "model_weights.pth")
        

t = Train()
t.train()
t.evaluate()
#new one
#
model = model.ImageClassifier().to(device)
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()
predictions = model(torch.randn(4, 3072).to(device)).argmax(dim=1)
print(predictions)
                
        
    
