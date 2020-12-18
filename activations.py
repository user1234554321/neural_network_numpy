import numpy as np

class Activation:

    def __init__(self,):
        pass
    def __call__(self, X):
        raise Exception('Not Implemented Error!')

    def forward(self, ):
        raise Exception('Not Implemented Error!')

    def backward(self, ):
        raise Exception('Not Implemented Error!')

class Relu(Activation):

    def __init__(self, ):
        super(Relu, self).__init__()

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.X = X.copy()
        self.activations = np.maximum(X, np.zeros_like(X))
        return self.activations

    def backward(self,):
        self.dactivations = self.activations.copy()
        self.dactivations[self.dactivations > 0] = 1.
        # print(self.dactivations.shape)
        return self.dactivations

class Sigmoid(Activation):

    def __init__(self, ):
        pass

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.X = X.copy()
        neg_exp = np.exp(-X)
        self.activations = 1. / (1. + neg_exp)
        return self.activations

    def backward(self, ):
        # print(id(self))
        # print('activations shape', self.activations.shape)
        return self.activations * (1.-self.activations)

class Linear(Activation):

    def __init__(self,):
        super(Linear, self).__init__()

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.X = X.copy()
        return self.X

    def backward(self,):
        return np.ones_like(self.X)

class Tanh(Activation):

    def __init__(self, eps=1e-6):
        super(Tanh, self).__init__()
        self.eps = eps

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.X = X.copy()
        self.activations = np.tanh(X)
        return self.activations

    def backward(self,):
        return 1./(np.cosh(self.X) + self.eps)

activation_fns = {'relu': Relu, 'sigmoid': Sigmoid, 'tanh': Tanh, 'linear': Linear}