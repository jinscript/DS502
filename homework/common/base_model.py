from abc import ABCMeta, abstractmethod
import numpy as np


class BaseModel:
    """ An abstract class for each model to extend
    """
    __metaclass__ = ABCMeta

    PHASE_TRAIN = "TRAIN"
    PHASE_TEST = "TEST"

    def __init__(self, **kwargs):
        """ Initialize model with any arguments. e.g. learning rate
        """
        self.max_iter = 100  # default model arguments
        for key, value in kwargs.iteritems():
            setattr(self, key, value)

    def fit(self, X_train, y_train):
        """ Given data and labels, learn model parameters
        """
        self.setup(X_train, y_train)
        X_train = self.preprocess_data(X_train, BaseModel.PHASE_TRAIN)
        for i in xrange(self.max_iter):
            y_hat = self.forward_pass(X_train)
            if self.stopping_criteria() == True:
                break
            self.backward_pass(X_train, y_train, y_hat)

    def predict(self, X_test):
        """ Given data predict labels
        """
        X_test = self.preprocess_data(X_test, BaseModel.PHASE_TEST)
        return self.forward_pass(X_test)

    def preprocess_data(self, X, phase):
        """ Override this method to do custom preprocessing for data
            here provides a default implementation for adding bias
        """
        return np.c_[X, np.ones((X.shape[0], 1))]

    def stopping_criteria(self):
        """ Override this method if you want to do early stopping
        """
        return False

    @abstractmethod
    def setup(self, X_train, y_train):
        """ Initialize the model, e.g. creating weight matrix
        """
        return

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
