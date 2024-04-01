from typing import Any
from layer import Layer

class MLP:
    def __init__(self, input_dim:int, layer_dims:list[int], activationfuncs:list[int], learning_rate=0.001):
        self.learning_rate = learning_rate
        self.network = []
        self.network.append(Layer(input_dim, layer_dims[0], activationfuncs[0]))
        for i in range(len(layer_dims)-1):
            self.network.append(Layer(layer_dims[i], layer_dims[i+1], activationfuncs[i]))
    
    def __call__(self, input):
        return self.feedforward(input)

    def feedforward(self, input):
        out = input
        for i in range(len(self.network)):
            out = self.network[i](out)       
        return out

    def backpropagation(self, ground_truth, output):
        error = output - ground_truth

        for i in reversed(range(len(self.network))):
            dW, dB, error = self.network[i].backpropagation(error)
            self.network[i].learn(dW, dB, self.learning_rate)
    