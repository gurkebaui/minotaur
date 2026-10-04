import torch

class Flattener:
    
    def __init__(self):
        self.tensor = torch.zeros((16, 3, 32, 32)) 

    def random(self):
        self.tensor = torch.rand(16, 3, 32, 32)
        return self.tensor
    def flatten(self):
        self.random()
        self.tensor_flat = self.tensor.flatten(start_dim=1)
        return  self.tensor_flat

class Unsqueeze:
    def __init__(self):
        self.v = torch.tensor([1.0, 2.0, 3.0, 4.0])

    def us(self, index):
        self.v = self.v.unsqueeze(index)
    def getShape(self):
        return self.v.size()




