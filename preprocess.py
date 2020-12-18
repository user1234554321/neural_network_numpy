import numpy as np 

class Scaler:

    def __init__(self, ):
        pass

    def fit(self, X):
        raise Exception('Not implemented error!')

    def fit_transform(self, X):
        raise Exception('Not implemented error!')

    def transform(self, X):
        raise Exception('Not implemented error')

class StandardScaler(Scaler):

    def __init__(self, eps=1e-6):
        super(StandardScaler, self).__init__()
        self.eps = eps

    def fit(self, X):
        self.means = np.mean(X, 0)
        self.stdevs = np.std(X, 0)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        X = X - self.means
        X /= (self.stdevs + self.eps)
        return X

class MinMaxScaler(Scaler):

    def __init__(self, ):
        super(MinMaxScaler, self).__init__()

    def fit(self, X):
        self.mins = np.min(X, 0)
        self.maxs = np.max(X, 0)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        X = (X - self.mins) / (self.maxs - self.mins)
        return X

class Sampler:

    def __init__(self, ):
        pass

    def fit(self, X):
        raise Exception('Not implemented error!')

    def sample(self, ):
        raise Exception('Not implemented error!')

class UnderSampler(Sampler):

    def __init__(self, ):
        super(UnderSampler, self).__init__()

    def fit(self, X, y):
        self.num_rows = len(X)
        self.X = X
        self.y_onehot = y
        self.y = np.argmax(y, 1)
        self.unique_classes, counts = np.unique(self.y, return_counts=True)
        self.class_probs = counts / self.num_rows
        self.row_probs = np.zeros(self.num_rows)
        for i, unique_class in enumerate(self.unique_classes):
            self.row_probs[self.y == unique_class] = self.class_probs[i]

    def sample(self, ):
        row_select = 1.-self.row_probs > np.random.rand(self.num_rows)
        return self.X[row_select], self.y_onehot[row_select]

class OverSampler(Sampler):

    def __init__(self, ):
        super(OverSampler, self).__init__()

    def fit(self, X, y):
        self.num_rows = len(X)
        self.X = X
        self.y_onehot = y
        self.y = np.argmax(y, 1)
        self.unique_classes, counts = np.unique(self.y, return_counts=True)
        self.multipliers = {class_label: multiplier+1 for class_label, multiplier in zip(self.unique_classes, (np.max(counts) / counts).astype('int'))}

    def sample(self, ):
        ind = int(np.random.rand() * self.num_rows)
        dataset_X = np.expand_dims(self.X[ind], 0)
        dataset_y = np.expand_dims(self.y_onehot[ind], 0)
        for ind, yi in enumerate(self.y):
            multiplier = int(np.random.rand() * self.multipliers[yi])
            if multiplier > 0:
                rows = np.repeat(np.expand_dims(self.X[ind], 0), multiplier, 0)
                labels = np.repeat(np.expand_dims(self.y_onehot[ind], 0), multiplier, 0)
                dataset_X = np.vstack((dataset_X, rows))
                dataset_y = np.vstack((dataset_y, labels))
        return dataset_X, dataset_y