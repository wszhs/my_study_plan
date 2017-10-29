
from numpy import *
import time
import matplotlib.pyplot as plt


def load_data(path):
    data_x = []
    data_y = []
    file_in = open(path)
    for line in file_in.readlines():
        line_arr = line.strip().split()
        data_x.append([1.0, int(line_arr[0]), int(line_arr[1])])
        data_y.append(int(line_arr[2]))
    return mat(data_x), mat(data_y).transpose()


# calculate the sigmoid function
def sigmoid(x):
    return 1.0 / (1 + exp(-x))


def train_log_regression(train_x, train_y, opts):
    # calculate training time
    start_time = time.time()
    num_samples, num_features = shape(train_x)
    alpha = opts['alpha'];
    max_iter = opts['maxIter']
    weights = zeros((num_features, 1))
    print weights
    for k in range(max_iter):
            output = sigmoid(train_x * weights)
            error = train_y - output
            weights = weights + alpha * train_x.transpose() * error

    print 'Congratulations, training complete! Took %fs!' % (time.time() - start_time)
    return weights


def test_log_regression(test_x, test_y, weights):
    num_samples, num_features = shape(test_x)
    match_count = 0
    for i in xrange(num_samples):
        predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
        if predict == bool(test_y[i, 0]):
            match_count += 1
    accuracy = float(match_count) / num_samples
    return accuracy


def show_log_regression(weights, train_x, train_y):
    print weights
    num_samples, num_features = shape(train_x)
    for i in xrange(num_samples):
        if int(train_y[i, 0]) == 0:
            plt.plot(train_x[i, 1], train_x[i, 2], 'or')
        elif int(train_y[i, 0]) == 1:
            plt.plot(train_x[i, 1], train_x[i, 2], 'ob')
            # draw the classify line
    min_x = min(train_x[:, 1])[0, 0]
    max_x = max(train_x[:, 1])[0, 0]
    weights = weights.getA()  # convert mat to array
    y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
    y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1');
    plt.ylabel('X2')
    plt.show()
