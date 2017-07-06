import numpy as np


class FunctionUtils:
    """ A collection of commonly used math functions
    """

    @staticmethod
    def sigmoid(x):
        """ y = 1 / (1 + e^-x)
        x: numpy 1d array
        return: numpy 1d array after transformation 
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def d_sigmoid(x):
        """ y = sigmoid(x) * (1 - sigmoid(x))
        x: numpy 1d array
        return: numpy 1d array after transformation
        """
        s = FunctionUtils.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def tanh(x):
        """ y = tanh(x)
        x: numpy 1d array
        return: numpy 1d array after transformation 
        """
        return np.tanh(x)

    @staticmethod
    def d_tanh(x):
        """ y = 1 / (cosh(x))^2
        x: numpy 1d array
        return: numpy 1d array after transformation
        """
        cosh = np.cosh(x)
        return 1 / np.square(cosh)

    @staticmethod
    def relu(x):
        """ y = x if x > 0 else 0
        x: numpy 1d array
        return: numpy 1d array after transformation 
        """
        return x * (x > 0)

    @staticmethod
    def d_relu(x):
        """ y = 1 if x > 0 else 0
        x: numpy 1d array
        return: numpy 1d array after transformation
        """
        return 1 * (x > 0)

    @staticmethod
    def softmax(x):
        """ y = e^(x_i - x_max) / sum(e^(x_i - x_max))
        Deduct x_max to prevent overflow
        x: numpy 1d array
        return: numpy 1d array after transformation
        """
        x = x - np.max(x)
        exp = np.exp(x)
        return exp / np.sum(exp)

    @staticmethod
    def d_softmax_cross_entropy(x, target):
        """ y = x - target
        x: numpy 1d array
        target: numpy 1d array
        return: numpy 1d array after transformation
        """
        return x - target

    @staticmethod
    def mean_squared_error(x, target):
        """ y = (x - target).T * (x - target) / n
        x: numpy 1d array
        target: numpy 1d array
        return: scalar
        """
        diff = x - target
        return np.dot(x.T, x) / x.shape[0]

    @staticmethod
    def accuracy(x, target):
        """ y = sum(x == target) / n
        x: numpy 1d array
        target: numpy 1d array
        return: double between 0 and 1
        """
        return np.sum(np.round(x) == target) / float(x.shape[0])

    @staticmethod
    def to_one_hot(x, n_class):
        """
        x: vector to convert to one hot representation
        n_class: one hot vector dimension
        """
        n = x.shape[0]
        one_hot = np.zeros((n, n_class))
        one_hot[np.arange(n), np.ravel(x).astype(int)] = 1
        return one_hot

    @staticmethod
    def from_one_hot(X):
        """
        X: one hot representation
        """
        return np.argmax(X, axis=1).reshape((X.shape[0], 1))
