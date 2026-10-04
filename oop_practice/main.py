class Neuron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def predictX(self,x):
        h = (x* self.weight) + self.bias
        return h

neuron = Neuron(1,2)
print(neuron.predictX(3))
