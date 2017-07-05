import sys
import os
splitted_dir = os.getcwd().split('/')
del splitted_dir[-1]
sys.path.append('/'.join(splitted_dir))

from common import BaseModel, FunctionUtils
from sklearn import datasets, linear_model
import numpy as np


class LinearRegression(BaseModel):

    def forward_pass(self, X):
        return np.dot(X, self.w_);

    def backward_pass(self, X, y, y_hat):
        """ w = (X.T * X + lambda * I)^-1 * X.T * Y
            Added l2 regularization
            Fastest for small data set
        """
        X_TX = np.dot(X.T, X) + np.eye(X.shape[1]) * self.lambda_
        X_Ty = np.dot(X.T, y)
        self.w_ = np.dot(np.linalg.inv(X_TX), X_Ty)

def main():
    # load dataset
    dataset = datasets.load_diabetes()
    # Select only 1 dimension
    X = dataset.data[:, 2]
    y = dataset.target

    # split dataset into training and testing
    X_train = X[:-20, None]
    X_test = X[-20:, None]

    y_train = y[:-20, None]
    y_test = y[-20:, None]

    linear_regression = LinearRegression(lambda_=0, w_=np.zeros((X_train.shape[1] + 1, 1)))
    linear_regression.fit(X_train, y_train, max_iter=1)

    print('Coefficients: {}'.format(linear_regression.w_[0 : -1]))
    print('Intercept: {}'.format(linear_regression.w_[-1]))
    print('MSE: {}'.format(FunctionUtils.mean_squared_error(linear_regression.predict(X_test), y_test)))

    # sklearn linear regression
    sklearn_linear_regression = linear_model.LinearRegression()
    sklearn_linear_regression.fit(X_train, y_train)

    print('Sklearn Coefficients: {}'.format(sklearn_linear_regression.coef_))
    print('Sklearn Intercept: {}'.format(sklearn_linear_regression.intercept_))
    print('Sklearn MSE: {}'.format(FunctionUtils.mean_squared_error(sklearn_linear_regression.predict(X_test), y_test)))

if __name__ == '__main__':
    main()
