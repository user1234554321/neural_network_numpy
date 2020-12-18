import numpy as np 

class Loss:

    def __init__(self, ):
        pass

class MSELoss(Loss):

    def __init__(self, eps=1e-6):
        super(MSELoss, self).__init__()
        self.eps = eps

    def __call__(self, y_pred, y):
        self.y_pred = y_pred
        self.y = y
        return np.sum((self.y - self.y_pred) ** 2) / y.shape[1]

    def backward(self, e):
        return -(self.y - self.y_pred)

    def __str__(self, ):
        return 'MSE'

class WeightedMSELoss(Loss):

    def __init__(self, weights, eps=1e-6):
        super(WeightedMSELoss, self).__init__()
        self.weights = np.expand_dims(np.array(weights), 1)
        self.eps = eps

    def __call__(self, y_pred, y):
        self.y_pred = y_pred
        self.y = y
        return np.sum((self.weights * (self.y - self.y_pred)) ** 2) / y.shape[1]

    def backward(self, e):
        return -self.weights * (self.y - self.y_pred) 

class SoftmaxCrossEntropyLoss(Loss):

    def __init__(self, eps=1e-6):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.eps = eps

    def __call__(self, y_pred, y):
        self.y_pred = self.softmax(y_pred)
        self.y = y
        return -np.sum(self.y * np.log(self.y_pred + self.eps)) / self.y.shape[1]
    
    def softmax(self, x):
        x = np.exp(x - np.max(x))
        denom = np.sum(x, 0)
        return x / denom

    def backward(self, e):
        return (self.y_pred - self.y)

    def __str__(self, ):
        return 'SoftmaxCrossEntropy'

class WeightedSoftmaxCrossEntropyLoss(Loss):

    def __init__(self, weights, eps=1e-6):
        super(WeightedSoftmaxCrossEntropyLoss, self).__init__()
        self.weights = np.expand_dims(np.array(weights), 1)
        print(self.weights)
        self.eps = eps

    def __call__(self, y_pred, y):
        self.y_pred = self.softmax(y_pred)
        self.y = y
        return -np.sum(self.y * self.weights * np.log(self.y_pred + self.eps)) / self.y.shape[1]
    
    def softmax(self, x):
        x = x - np.max(x)
        denom = np.sum(np.exp(x))
        return np.exp(x) / denom

    def backward(self, e):
        return (self.y_pred - self.weights * self.y) 
