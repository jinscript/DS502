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

    ACTIVATION_SIGMOID = 'sigmoid'
    ACTIVATION_TANH = 'tanh'
    ACTIVATION_RELU = 'relu'

    def setup(self, X_train, y_train):

        np.random.seed(0)
        self.stop_count = 0
        self.stop_count_threshold = 3

        n_input = X_train.shape[1]
        n_output = y_train.shape[1]

        # initialize weights
        self.w_ = []
        self.w_.append(np.random.uniform(-1, 1, size=(n_input + 1, self.hidden_layer_sizes[0])))

        for i in xrange(1, len(self.hidden_layer_sizes) - 1):
            self.w_.append(np.random.uniform(-1, 1, size=(self.hidden_layer_sizes[i] + 1, self.hidden_layer_sizes[i + 1] + 1)))

        self.w_.append(np.random.uniform(-1, 1, size=(self.hidden_layer_sizes[-1] + 1, n_output)))

        self.h_ = []
        self.curr_accuracy = 0

        if self.activation == MLP.ACTIVATION_SIGMOID:
            self.activation_func = FunctionUtils.sigmoid
            self.d_activation_func = FunctionUtils.d_sigmoid
        elif self.activation == MLP.ACTIVATION_TANH:
            self.activation_func = FunctionUtils.tanh
            self.d_activation_func = FunctionUtils.d_tanh
        elif self.activation == MLP.ACTIVATION_RELU:
            self.activation_func = FunctionUtils.relu
            self.d_activation_func = FunctionUtils.d_relu
        else:
            raise Exception('Unsupported activation')

    def forward_pass(self, X):

        del self.h_[:]
        n_train = X.shape[0]
        h = np.c_[self.activation_func(np.dot(X, self.w_[0])), np.ones((n_train, 1))]
        self.h_.append(h)
        for i in xrange(1, len(self.w_) - 1):
            h = np.c_[self.activation_func(np.dot(h, self.w_[i])), np.ones((n_train, 1))]
            self.h_.append(h)
        return FunctionUtils.softmax(np.dot(h, self.w_[-1]))

    def backward_pass(self, X, y, y_hat):

        n_train = X.shape[0]

        d_l_d_output = FunctionUtils.d_softmax_cross_entropy(y_hat, y)
        d_w = np.dot(self.h_[-1].T, d_l_d_output)
        d_h = np.dot(d_l_d_output, self.w_[-1].T)[: , :-1]  # discard gradient for bias
        self.w_[-1] -= self.lr_ / n_train * d_w

        for i in xrange(len(self.w_) - 2, 0):
            d_wh = self.d_activation_func(d_h)
            d_w = np.dot(self.h[i].T, d_wh)
            d_h = np.dot(d_wh, self.w_[i].T)[: , :-1]
            self.w_[i] -= self.lr_ / n_train * d_w

        d_wX = self.d_activation_func(d_h)
        d_w = np.dot(X.T, d_wX)
        self.w_[0] -= self.lr_ / n_train * d_w * 1.6

    def preprocess_data(self, X, phase):

        if (phase == BaseModel.PHASE_TRAIN):
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
            self.X_std[self.X_std == 0] = 1  # avoid divide by 0
        return np.c_[(X - self.X_mean) / self.X_std, np.ones((X.shape[0], 1))]

    def stop_criteria(self, X_train, y_train, y_hat):

        accuracy = FunctionUtils.accuracy(FunctionUtils.from_one_hot(y_train), FunctionUtils.from_one_hot(y_hat))
        if (accuracy - self.curr_accuracy < self.tol):
           self.stop_count += 1
           return self.stop_count >= self.stop_count_threshold
        self.stop_count = 0
        self.curr_accuracy = accuracy
        return False

def run_digits():
    print 'Using digits dataset'
    dataset = datasets.load_digits()

    # Use all the features
    X = dataset.data[:, :]
    y = dataset.target[:, None]

    n_class = len(np.unique(y))

    X_train = X[:1500, :]
    y_train = y[:1500, :]

    X_test = X[1500:, :]
    y_test = y[1500:, :]

    # mlp
    mlp = MLP(hidden_layer_sizes=(128,), max_iter=5000, lr_=0.1, activation=MLP.ACTIVATION_SIGMOID, tol=1e-6)
    mlp.fit(X_train, FunctionUtils.to_one_hot(y_train, n_class))
    y_train_pred = FunctionUtils.from_one_hot(mlp.predict(X_train))
    y_test_pred = FunctionUtils.from_one_hot(mlp.predict(X_test))
    print('mlp training set accuracy: {}'.format(FunctionUtils.accuracy(y_train_pred, y_train)))
    print('mlp test set accuracy: {}'.format(FunctionUtils.accuracy(y_test_pred, y_test)))

    # sklearn mlp
    sklearn_mlp = MLPClassifier(hidden_layer_sizes=(128), max_iter=200, alpha=1e-4,
                        solver='sgd', activation='logistic', verbose=False, tol=1e-4, random_state=1,
                        learning_rate_init=0.01)
    sklearn_mlp.fit(X_train, np.ravel(y_train))
    print('sklearn training set accuracy: {}'.format(sklearn_mlp.score(X_train, np.ravel(y_train))))
    print('sklearn test set score: {}'.format(sklearn_mlp.score(X_test, np.ravel(y_test))))

def run_mnist():
    print 'Using MNIST dataset'
    dataset = datasets.fetch_mldata("MNIST original")

    # Use all the features
    X = dataset.data[:, :]
    y = dataset.target[:, None]

    n_class = len(np.unique(y))

    X_train = X[:60000, :]
    y_train = y[:60000, :]

    X_test = X[60000:, :]
    y_test = y[60000:, :]

    # mlp
    mlp = MLP(hidden_layer_sizes=(128,), max_iter=5000, lr_=0.1, activation=MLP.ACTIVATION_SIGMOID, tol=1e-4)
    mlp.fit(X_train, FunctionUtils.to_one_hot(y_train, n_class))
    y_train_pred = FunctionUtils.from_one_hot(mlp.predict(X_train))
    y_test_pred = FunctionUtils.from_one_hot(mlp.predict(X_test))
    print('mlp training set accuracy: {}'.format(FunctionUtils.accuracy(y_train_pred, y_train)))
    print('mlp test set accuracy: {}'.format(FunctionUtils.accuracy(y_test_pred, y_test)))

    # sklearn mlp
    sklearn_mlp = MLPClassifier(hidden_layer_sizes=(128), max_iter=200, alpha=1e-4,
                        solver='sgd', activation='logistic', verbose=False, tol=1e-4, random_state=1,
                        learning_rate_init=0.01)
    sklearn_mlp.fit(X_train, np.ravel(y_train))
    print('sklearn training set accuracy: {}'.format(sklearn_mlp.score(X_train, np.ravel(y_train))))
    print('sklearn test set score: {}'.format(sklearn_mlp.score(X_test, np.ravel(y_test))))

def main():
    run_digits()
    # run_mnist()

if __name__ == '__main__':
    main()
