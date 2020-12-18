import numpy as np
from activations import activation_fns, Relu, Linear, Sigmoid, Tanh

class Layer:

    def __init__(self,):
        pass

    def forward(self, ):
        raise Exception('Not Implemented Error!')

    def backward(self, ):
        raise Exception('Not Implemented Error!')

class FeedForwardLayer(Layer):

    def __init__(self, name='', input_size=10, output_size=20, initializer='random', activation_fn='linear'):
        super(FeedForwardLayer, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        if initializer == 'random':
            self._w = np.random.randn(output_size, input_size)
        elif initializer == 'zeros':
            self._w = np.zeros((output_size, input_size))
        elif initializer == 'glorot':
            self._w = np.random.randn(output_size, input_size)*np.sqrt(2/(input_size + output_size))
        self._b = np.zeros((output_size, 1))
        self._wgrad = np.zeros_like(self._w)
        self._bgrad = np.zeros_like(self._b)
        self._dw = np.zeros_like(self._w)
        self._db = np.zeros_like(self._b)
        self.activation_fn = activation_fns[activation_fn]()

    def _zero_grad(self, ):
        self._wgrad = np.zeros_like(self._w)
        self._bgrad = np.zeros_like(self._b)

    def forward(self, X):
        assert X.shape[0] == self._input_size
        self.X = X.copy()
        self.Z = np.dot(self._w, X) + self._b
        self.A = self.activation_fn(self.Z)
        return self.A.copy()

    def backward(self, da):
        dz = da * self.activation_fn.backward() 
        self._wgrad = np.dot(dz, self.X.T)
        self._bgrad = np.expand_dims(dz.sum(1), 1)
        da = np.dot(self._w.T, dz)
        return da
