import sys
import os
splitted_dir = os.getcwd().split('/')
del splitted_dir[-1]
sys.path.append('/'.join(splitted_dir))

from common import BaseModel, FunctionUtils
from sklearn import datasets, linear_model
import numpy as np


class LogisticRegression(BaseModel):

    def forward_pass(self, X):
        return FunctionUtils.sigmoid(np.dot(X, self.w_));

    def backward_pass(self, X, y, y_hat):
        """ Batch gradient descent
            g = X.T * (sigmoid(X * w) - y)
        """
        self.w_ -= self.lr_ / X.shape[0] * (np.dot(X.T, (y_hat - y)) + self.lambda_ * self.w_)

    def prepare(self, X):
        """ Normalize data and add bias
        """
        return np.c_[(X - self.X_mean) / self.X_std, np.ones((X.shape[0], 1))]

def main():
    # load dataset
    dataset = datasets.load_iris()
    # Select only 2 dims
    X = dataset.data[0:100, 0:2]
    y = dataset.target[:100, None]

    # split dataset into training and testing
    idx_train = range(30)
    idx_train.extend(range(50, 80))
    idx_test = range(30,50)
    idx_test.extend(range(80, 100))

    X_train = X[idx_train]
    X_test = X[idx_test]

    y_train = y[idx_train]
    y_test = y[idx_test]

    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)

    logistic_regression = LogisticRegression(lambda_=0, lr_=0.05, w_=np.zeros((X_train.shape[1] + 1, 1)),
                                             X_mean=X_mean, X_std=X_std)
    logistic_regression.fit(X_train, y_train, max_iter=50000)

    print('Coefficients: {}'.format(logistic_regression.w_[0 : -1].T))
    print('Intercept: {}'.format(logistic_regression.w_[-1]))
    print('Accuracy: {}'.format(FunctionUtils.accuracy(logistic_regression.predict(X_test), y_test)))

    # sklearn logistic regression
    sklearn_logistic_regression = linear_model.LogisticRegression()
    sklearn_logistic_regression.fit(X_train, np.ravel(y_train))

    print('Sklearn Coefficients: {}'.format(sklearn_logistic_regression.coef_))
    print('Sklearn Intercept: {}'.format(sklearn_logistic_regression.intercept_))
    print('Sklearn Accuracy: {}'.format(FunctionUtils.accuracy(sklearn_logistic_regression.predict(X_test), np.ravel(y_test))))

if __name__ == '__main__':
    main()
