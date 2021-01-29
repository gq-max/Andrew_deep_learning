import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
import PIL
from scipy import ndimage
import scipy.misc
from lr_utils import load_dataset

# train_set_x_orig = (m_train=209, dim=64, dim=64, 3)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# # 列子
# index = 5
# plt.imshow(train_set_x_orig[index])
# print("y = " +
#       str(train_set_y[:, index]) +
#       ", it's a '" +
#       classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.")
# plt.show()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]  # 50
num_px = train_set_x_orig.shape[1]

# 转换训练集和测试集(12288, 209)和(12288, 50)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# 数据预处理
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def init_parameter(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))  # compute cost

    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)

    cost = np.squeeze(cost)
    grads = {"dw": dw, "db": db}
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w, "b": b}

    grads = {"dw": dw, "db": db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):

        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = init_parameter(X_train.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train,
                                        num_iterations, learning_rate, print_cost)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


d = model(train_set_x, train_set_y, test_set_x, test_set_y,
          num_iterations=2000, learning_rate=0.005, print_cost=True)
index = 1
# plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
print("y = " + str(test_set_y[0, index]) +
      ", you predicted that it is a \"" +
      classes[int(d["Y_prediction_test"][0, index])].decode("utf-8") + "\" picture.")
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
