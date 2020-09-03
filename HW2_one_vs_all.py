from sklearn.preprocessing import label_binarize
import os
from tensorflow.keras.datasets import mnist
import numpy as np
import timeit

# loading data
current_dir = os.getcwd()
(X_train, Y_train), (X_test, Y_test) = mnist.load_data(path=current_dir + '/mnist.npz')

# RESHAPING DATA
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
Y_train = Y_train.reshape(60000, 1)
Y_test = Y_test.reshape(10000, 1)

print('MNIST Dataset Shape:')
print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(Y_train.shape))
print('X_test:  ' + str(X_test.shape))
print('Y_test:  ' + str(Y_test.shape))


def binarizer(C, y):
    binary_y = np.zeros((y.size, 1))
    for i in range(y.size):
        if y[i] == C:
            binary_y[i] = 1
        else:
            binary_y[i] = 0
    # print("Class Label: ", C)
    # print("Y train Value: ", y[2])
    # print("Binary Value: ", binary_y[2])
    return binary_y


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def gradient_descent(y, h, x):
    #print(loss[2])
    grad = (np.dot(np.transpose(h - y), x)) / y.size
    return grad


# INIT PARAMS
lr = 0.0001
class_preds = np.zeros((10, 10000))

# TRAIN EACH CLASS
start = timeit.default_timer()

for k in range(10):
    class_label = k
    Y_train_binary = binarizer(class_label, Y_train)
    W = np.zeros((1, 784))

    for i in range(50):
        for j in range(600):
            X_train_batch = X_train[(j * 100 + 0):(j + 1) * 100, ]
            Y_train_batch = Y_train_binary[(j * 100 + 0):(j + 1) * 100, ]
            Z = np.dot(X_train_batch, W.transpose())
            h = sigmoid(Z)
            #print(h[2])
            step = gradient_descent(Y_train_batch, h, X_train_batch)
            W = W - lr * step

    Y_test_binary = binarizer(class_label, Y_test)
    Z = np.dot(X_test, W.transpose())
    y_pred = sigmoid(Z)

    y_pred = y_pred.reshape(10000)

    class_preds[k] = y_pred

stop = timeit.default_timer()

class_preds = np.transpose(class_preds)
preds = class_preds.argmax(axis=1)

correct = 0
total = 10000
for i in range(total):
    for k in range(10):
        if Y_test[i] == k and preds[i] >= 0.5:  # index
            correct += 1
        #if Y_test[i] != k and y_pred[i] < 0.5:  # index
            #correct += 1

accuracy = (correct / total)
print("ACCURANCY OF ONE VS. ALL TRAINING IS: ", accuracy)

print('Time: ', stop - start)
