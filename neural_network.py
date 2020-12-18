import numpy as np 
from layers import FeedForwardLayer
from losses import MSELoss, SoftmaxCrossEntropyLoss, WeightedSoftmaxCrossEntropyLoss, WeightedMSELoss

class NeuralNetwork:

    def __init__(self, ):
        self.layers = []


    def add_layer(self, layer_name='', input_size=None, units=10, activation_fn='linear', initializer='random'):
        if self.layers == [] and input_size is None:
            raise Exception('First layer must have input shape!')
        if self.layers == []:
            layer = FeedForwardLayer(input_size=input_size, output_size=units, activation_fn=activation_fn, initializer=initializer)
        else:
            layer = FeedForwardLayer(input_size=self.layers[-1]._output_size, output_size=units, activation_fn=activation_fn,  initializer=initializer)
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, e):
        for layer in self.layers[::-1]:
            e = layer.backward(e)

    
