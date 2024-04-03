import numpy as np

class Layer:
    def __init__(self,features,neurons,activationfunc):
        self.features            = features
        self.neurons             = neurons
        if activationfunc not in ['linear','relu','softmax']:
            raise ValueError("Wrong activation functions value")
        self.activationfunc      = activationfunc
        self.weight              = np.random.randn(self.neurons, self.features)
        self.bias                = np.random.randn(neurons,1)
        self.input               = np.zeros([self.features,1])
        self.output              = np.zeros([self.features,1])

    def __call__(self, input):
        return self.feedforward(input)

    def activationReLu(self,inputs):
        return np.maximum(inputs,0)
        
    def derivativeReLu(self,inputs):
        return inputs > 0
    
    def activationSoftMax(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs))
        return exp_values / np.sum(exp_values)
        
    def derivativeSofMax(self,input):
        return self.activationSoftMax(input)*(1-self.activationSoftMax(input))
    
    def derivativeFunction(self,input):
        if self.activationfunc   == "relu":
            return self.derivativeReLu(input)
        elif self.activationfunc == "softmax":
            return self.derivativeSofMax(input)
        elif self.activationfunc == 'linear' :
            return 1
        else:
            raise ValueError("Wrong activation functions value")
    
    def activationFunction(self,inputs):
        if self.activationfunc   == "relu":
            return self.activationReLu(inputs)
        elif self.activationfunc == "softmax":
            return self.activationSoftMax(inputs)
        elif self.activationfunc == 'linear' :
            return inputs
        else:
            raise ValueError("Wrong activation functions value")


    def feedforward(self,input):
        self.input = input
        self.output      = np.dot(self.weight, input)
        self.output      += self.bias
        activation  = self.activationFunction(self.output)  
        return activation
        
    def backpropagation(self,dZ):
        m    = dZ.shape[1]
        dEdZ = dZ*self.derivativeFunction(self.output)
        dW   = (1/m) * np.dot(dEdZ, self.input.transpose())
        dB   = (1/m) * np.sum(dEdZ, axis=1, keepdims=True)
        dE   = np.dot(self.weight.transpose(), dEdZ)
 
        return dW, dB, dE

    def learn(self, dW, dB, learning_rate):
        self.weight -= dW*learning_rate
        self.bias   -= dB*learning_rate

    def __repr__(self) -> str:
        return (str("Layer input dim: "+str(self.features))+
                str("Layer output dim: "+str(self.neurons))+
                str("Layer activation function: "+str(self.features)))