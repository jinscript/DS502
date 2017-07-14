import sys
import os
splitted_dir = os.getcwd().split('/')
del splitted_dir[-1]
sys.path.append('/'.join(splitted_dir))

from common import BaseModel, FunctionUtils
from sklearn import datasets
import numpy as np


class SoftmaxRegression(BaseModel):
    
    def setup(self, X_train, y_train):
        self.w_ = np.zeros((X_train.shape[1] + 1, y_train.shape[1]))

    def forward_pass(self, X):
        return FunctionUtils.softmax(np.dot(X, self.w_));

    def backward_pass(self, X, y, y_hat):
        """ g = X.T * (y_hat - y)
        """
        self.w_ -= self.lr_ / X.shape[0] * np.dot(X.T, FunctionUtils.d_softmax_cross_entropy(y_hat, y))

    def preprocess_data(self, X, phase):
        """ Normalize data and add bias
        """
        if (phase == BaseModel.PHASE_TRAIN):
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
            self.X_std[self.X_std == 0] = 1  # avoid divide by 0

        return np.c_[(X - self.X_mean) / self.X_std, np.ones((X.shape[0], 1))]

def main():
    # Load the digits dataset
    dataset = datasets.load_digits()

    # Use all the features
    X = dataset.data[:, :]
    y = dataset.target[:, None]

    n_class = len(np.unique(y))

    X_train = X[:-100, :]
    y_train = y[:-100, :]

    X_test = X[-100:, :]
    y_test = y[-100:, :]

    softmax_regression = SoftmaxRegression(lr_=0.05, max_iter=500)
    softmax_regression.fit(X_train, FunctionUtils.to_one_hot(y_train, n_class))
    y_pred = FunctionUtils.from_one_hot(softmax_regression.predict(X_test))
    print('Accuracy: {}'.format(FunctionUtils.accuracy(y_pred, y_test)))

if __name__ == "__main__":
    main()
