from abc import ABCMeta, abstractmethod
import numpy as np


class BaseModel:
    """ An abstract class for each model to extend
    """
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        """ Initialize model with any arguments. e.g. parameters
        """
        for key, value in kwargs.iteritems():
            setattr(self, key, value)

    def fit(self, X_train, y_train, max_iter=100):
        """ Given data and labels, learn model parameters
        """
        X_train = self.prepare(X_train)
        for i in xrange(max_iter):
            y_hat = self.forward_pass(X_train)
            self.backward_pass(X_train, y_train, y_hat)

    def predict(self, X_test):
        """ Given data predict labels
        """
        X_test = self.prepare(X_test)
        return self.forward_pass(X_test)

    def prepare(self, X):
        """ Override this method to do custom prepration for data
            here provides a default implementation for adding bias
        """
        return np.c_[X, np.ones((X.shape[0], 1))]

    @abstractmethod
    def forward_pass(self, X):
        """ Given data predict labels
        """
        return

    @abstractmethod
    def backward_pass(self, X, y, y_hat):
        """ Given data, label, predictions, update model parameters
        """
        return
