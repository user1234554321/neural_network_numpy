import numpy as np
from losses import MSELoss, SoftmaxCrossEntropyLoss

class Optimizer:

    def __init__(self, model, criterion=None, learning_rate=1e-5, alpha=0.0, mom=0.0):

        self.learning_rate = learning_rate
        self.model = model
        self.alpha = alpha
        self.mom = mom
        if criterion is None:
            self.criterion = SoftmaxCrossEntropyLoss()
        else:
            self.criterion = criterion
        self.epoch = 0
        self.epsilon = 1e-6

    def _update_weights(self, batch_size):
        raise Exception('Not implemented error!')

    def _get_minibatch(self, X, y, mini_batch_size):
        length = X.shape[0]
        for i in range(0, length, mini_batch_size):
            X_batch = X[i * length: (i+1) * length]
            y_batch = y[i * length: (i+1) * length]
            yield (X_batch, y_batch)

    def fit_batch(self, X, y, epochs=1):

        num_samples = len(X)
        
        for epoch in range(epochs):
            self.epoch += 1

            y_pred = self.model.forward(X.T)
            
            loss = self.criterion(y_pred, y.T)

            e = self.criterion.backward(loss)

            self.model.backward(e)

            self._update_weights(batch_size=num_samples)


    def fit_minibatch(self, X, y, mini_batch_size=16, epochs=1):
        
        for epoch in range(epochs):
            self.epoch += 1
            for (X_batch, y_batch) in self._get_minibatch(X, y, mini_batch_size):
                
                y_pred = self.model.forward(X.T)
                
                loss = self.criterion(y_pred, y.T)

                e = self.criterion.backward(loss)

                self.model.backward(e)

                self._update_weights(batch_size=mini_batch_size)

    def fit_stochastic(self, X, y, epochs=1):
        
        for epoch in range(epochs):
            self.epoch += 1
            for (xi, yi) in zip(X, y):
                
                y_pred = self.model.forward(np.expand_dims(xi, 1))
                
                loss = self.criterion(y_pred, np.expand_dims(yi, 1))

                e = self.criterion.backward(loss)

                self.model.backward(e)

                self._update_weights(batch_size=1)

class SGDOptimizer(Optimizer):
       
        def __init__(self, model, criterion=None, learning_rate=1e-5, alpha=0.0, mom=0.0):
            super(SGDOptimizer, self).__init__(model, criterion, learning_rate, alpha, mom)
    
        def _update_weights(self, batch_size):

            for layer in self.model.layers:
                layer._wgrad /= batch_size
                layer._bgrad /= batch_size

                layer._dw = layer._dw * self.mom + layer._wgrad
                layer._db = layer._db * self.mom + layer._bgrad

                layer._w = layer._w - self.learning_rate * layer._dw - self.alpha * layer._w
                layer._b = layer._b - self.learning_rate * layer._db - self.alpha * layer._b

        def __str__(self, ):
            return 'SGDOptimizer'

class AdamOptimizer(Optimizer):
       
        def __init__(self, model, criterion=None, learning_rate=1e-5, alpha=0.0, mom=0.9, beta=0.999):
            super(AdamOptimizer, self).__init__(model, criterion, learning_rate, alpha, mom)
            self.beta = beta
            self.t = 1
            for layer in self.model.layers:
                layer._vdw = np.zeros_like(layer._w)
                layer._vdb = np.zeros_like(layer._b)

        def _update_weights(self, batch_size):

            self.t += 1
            for layer in self.model.layers:
                #Averaging gradients
                layer._wgrad /= batch_size
                layer._bgrad /= batch_size
                #Adaptive gradients
                layer._vdw = (self.beta * layer._vdw + (1.-self.beta) * np.power(layer._wgrad, 2))
                layer._vdb = (self.beta * layer._vdb + (1.-self.beta) * np.power(layer._bgrad, 2))

                vdw_hat = layer._vdw / (1.-np.power(self.beta, self.t))
                vdb_hat = layer._vdb / (1.-np.power(self.beta, self.t))
                #Momentum
                layer._dw = (layer._dw * self.mom + (1.-self.mom) * layer._wgrad)
                layer._db = (layer._db * self.mom + (1.-self.mom) * layer._bgrad)

                dw_hat = layer._dw / (1.-np.power(self.mom, self.t))
                db_hat = layer._db / (1.-np.power(self.mom, self.t))


                layer._w = layer._w - self.learning_rate * dw_hat/ (np.sqrt(vdw_hat) + self.epsilon) - self.alpha * layer._w
                layer._b = layer._b - self.learning_rate * db_hat / (np.sqrt(vdb_hat) + self.epsilon) - self.alpha * layer._b
                
                # layer._w = layer._w - self.learning_rate * layer._dw - self.alpha * layer._w
                # layer._b = layer._b - self.learning_rate * layer._db - self.alpha * layer._b
                
                # layer._zero_grad()
    
        def __str__(self, ):
            return 'AdamOptimizer'