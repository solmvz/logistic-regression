import sklearn
import os
import numpy as np
import matplotlib.pyplot as plt
import timeit

current_dir = os.getcwd()
from tensorflow.keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data(path=current_dir + '/mnist.npz')

# FLATTING TRAIN DATA
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# ONE HOT ENCODING
nb_classes = 10
targets1 = np.array(Y_train).reshape(-1)
Y_train = np.eye(nb_classes)[targets1]
targets2 = np.array(Y_test).reshape(-1)
Y_test = np.eye(nb_classes)[targets2]

# DATASET SHAPE
print('MNIST Dataset Shape:')
print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(Y_train.shape))
print('X_test:  ' + str(X_test.shape))
print('Y_test:  ' + str(Y_test.shape))


# SOFTMAX ACTIVATION FUNCTION
def activation_function(z):
    """returns y hat"""
    a = np.transpose(np.transpose(np.exp(z)) / (np.sum(np.exp(z), axis=1)))
    return a


def loss_function(y, y_hat):
    L = -1 * np.sum(np.matmul(np.transpose(y), np.log(y_hat)))
    return L


def gradient_descent(y, y_hat, x):
    grad = np.transpose(np.dot(np.transpose(x), (y - y_hat))) / y.size
    # grad = (y - y_hat)*0.001
    # print(grad[2])
    return grad

# INIT PARAMS
W = np.zeros((10, 784))
b = np.zeros(10)
y_hat = np.zeros(10)

learningRate = 0.001

# TRAIN
start = timeit.default_timer()

for i in range(50):
    for j in range(100):
        X_train_batch = X_train[(j * 600 + 0):(j + 1) * 600, ]
        Y_train_batch = Y_train[(j * 600 + 0):(j + 1) * 600, ]
        Z = np.dot(X_train_batch, W.transpose())
        y_hat = activation_function(Z)
        step = gradient_descent(Y_train_batch, y_hat, X_train_batch)
        W = W + learningRate * step

stop = timeit.default_timer()

# EVALUATE
Z = np.dot(X_test, W.transpose())
y_hat = activation_function(Z)
y_pred = y_hat.argmax(axis=1)
true_labels = Y_test.argmax(axis=1)

correct = 0
total = 10000
for i in range(total):
    if y_pred[i] == true_labels[i]:  # index
        correct += 1

accuracy = (correct / total)
print("ACCURANCY OF SOFTMAX TRAINING IS: ", accuracy)

print('Time: ', stop - start)

