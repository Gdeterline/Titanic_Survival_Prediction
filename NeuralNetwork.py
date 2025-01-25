import torch
from torch.nn import Module, Linear, ReLU, Sigmoid
from torch.optim import Adam, SGD

class NeuralNetwork(Module):
    def __init__(self, neurons):
        super(NeuralNetwork, self).__init__()
        # Init of our parameters
        self.neurons = neurons
        self.layers = torch.nn.ModuleList()
        for i in range(len(neurons)-1):
            self.layers.append(Linear(in_features=neurons[i], out_features=neurons[i+1]))
    
    def forward(self, x):
        for i in range(len(self.neurons)-2):
            x = self.layers[i](x)
            x = torch.relu(x)
        x = self.layers[-1](x)
        x = torch.sigmoid(x)
        return x
    