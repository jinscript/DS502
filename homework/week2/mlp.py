import sys
import os
splitted_dir = os.getcwd().split('/')
del splitted_dir[-1]
sys.path.append('/'.join(splitted_dir))

from common import BaseModel, FunctionUtils
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
import numpy as np


class MLP(BaseModel):

    def forward_pass(self, X):
        pass

    def backward_pass(self, X, y, y_hat):
        pass

def main():
    # digits dataset
    dataset = datasets.load_digits()

    # Use all the features
    X = dataset.data[:, :]
    y = dataset.target[:, None]

    n_class = len(np.unique(y))

    X_train = X[:1500, :]
    y_train = y[:1500, :]

    X_test = X[1500:, :]
    y_test = y[1500:, :]

    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std == 0] = 1  # avoid divide by 0

    # mlp

    # sklearn mlp
    sklearn_mlp = MLPClassifier(hidden_layer_sizes=(128), max_iter=200, alpha=1e-4,
                        solver='sgd', activation='logistic', verbose=False, tol=1e-4, random_state=1,
                        learning_rate_init=.01)
    sklearn_mlp.fit(X_train, np.ravel(y_train))
    print("sklearn training set accuracy: %f" % sklearn_mlp.score(X_train, np.ravel(y_train)))
    print("sklearn test set score: %f" % sklearn_mlp.score(X_test, np.ravel(y_test)))

    # MNIST dataset



if __name__ == '__main__':
    main()
