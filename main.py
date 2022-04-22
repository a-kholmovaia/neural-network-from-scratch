import pandas as pd
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator / denominator
    return softmax


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Network:
    def __init__(self, sizes, epochs=10, l_rate=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate
        self.params = self.initialization()

    def initialization(self):
        input_layer = self.sizes[0]
        hidden_1 = self.sizes[1]
        hidden_2 = self.sizes[2]
        output_layer = self.sizes[3]

        params = {
            'W1': np.random.rand(input_layer, hidden_1),
            'W2': np.random.rand(hidden_1, hidden_2),
            'W3': np.random.rand(hidden_2, output_layer),
            'B1': np.zeros(shape=(hidden_1, 1)),
            'B2': np.zeros(shape=(hidden_2, 1)),
            'B3': np.zeros(shape=(output_layer, 1))
        }

        return params

    def feed_forward(self, X_train):
        pms = self.params
        pms['A0'] = X_train

        # Layer 1
        pms['Z1'] = pms['B1'].T + np.dot(X_train, pms['W1'])
        pms['A1'] = sigmoid(pms['Z1'])

        # Layer 2
        pms['Z2'] = pms['B2'].T + np.dot(pms['A1'], pms['W2'])
        pms['A2'] = sigmoid(pms['Z2'])

        # Output Layer
        pms['Z3'] = pms['B3'].T + np.dot(pms['A2'], pms['W3'])
        pms['A3'] = softmax(pms['Z3'])

    def back_propagation(self, y):
        pms = self.params

        # Output Layer
        delta3 = pms['A3'] - y
        pms['W3'] -= self.l_rate * np.dot(pms['A2'].T, delta3)
        pms['B3'] -= self.l_rate * delta3.T

        # Layer 2
        delta2 = np.dot(delta3, pms['W3'].T)
        pms['W2'] -= self.l_rate * np.dot(pms['A1'].T, delta2)
        pms['B2'] -= self.l_rate * delta2.T

        # Layer 1
        delta1 = np.dot(delta2, pms['W2'].T)
        pms['W1'] -= self.l_rate * np.dot(pms['A0'].reshape(784, 1), delta1)
        pms['B1'] -= self.l_rate * delta1.T

    def accuracy_score(self, p, y):
        pred = []
        for p, y in zip(p, y):
            pred.append(p == y)

        return np.mean(pred)

    def train(self, X_train, y_train, X_val, y_val):
        for iteration in range(self.epochs):
            for x, y in zip(X_train, y_train):
                self.feed_forward(x)
                self.back_propagation(y)

            pred = self.predict(X_val)
            accuracy = self.accuracy_score(pred, y_val)
            print('Epoch: {}, Accuracy: {}'.format(iteration + 1, accuracy))

    def predict(self, X):
        pred = []

        for x in X:
            self.feed_forward(x)
            answer = np.argmax(self.params['A3'])
            pred.append(answer)

        return pred


if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape(train_X.shape[0], 28 * 28)
    test_X = test_X.reshape(test_X.shape[0], 28 * 28)
    X_train = (train_X / 255).astype('float32')
    X_test = (test_X / 255).astype('float32')
    y_train = to_categorical(train_y)
    y_test = to_categorical(test_y)
    x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=77)
    print("loading - done")

    dnn = Network(sizes=[784, 64, 32, 10])
    dnn.train(x_train, y_train, x_val, y_val)

    pred = dnn.predict(X_test)
    print("Final accuracy: {}".format(dnn.accuracy_score(pred, y_test)))
