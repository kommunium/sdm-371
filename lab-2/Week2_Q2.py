# %% import package and load data
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time

data = load_iris()
iris_features = pd.DataFrame(data=data.data, columns=data.feature_names)
X = np.array(iris_features['sepal length (cm)'])
Y = np.array(iris_features['petal length (cm)'])
m = len(X)  # size of the dataset

plt.scatter(X, Y)
plt.ylabel('sepal length (cm)')
plt.xlabel('petal length (cm)')
plt.show()

# %% initialize the model
learning_rate = 1e-3
theta_1 = 0
theta_0 = 0
history = []

# %% iteration
start = time.time()
# num_iterations = 100
num_iterations = 100

for i in range(num_iterations):
    Y_hat = theta_1 * X + theta_0  # forward prop
    distance = Y_hat - Y  # intermediate variable
    cost = (distance ** 2).sum() / 2 / m  # cost function
    history.append(cost)  # record cost history
    d1, d0 = np.dot(distance, X) / m, distance.mean()  # gradient

    # update parameter
    theta_1 -= learning_rate * d1
    theta_0 -= learning_rate * d0

end = time.time()
print('Iterated', num_iterations, 'times in', end - start, 'seconds')

plt.plot(history)
plt.title('Cost History')
plt.ylabel('Cost')
plt.xlabel('Iteration')
# plt.savefig('cost.png')
plt.show()

# %% regression result

plt.title('The result of the linear regression')
plt.grid()
plt.scatter(X, Y, color='C0')
plt.plot(X, theta_1 * X + theta_0, color='C1')
plt.ylabel('sepal length (cm)')
plt.xlabel('petal length (cm)')
# plt.savefig('result.png')
plt.show()

# %% performance analysis
# def iterate(num_iterations, theta_1, theta_0):
#     for i in range(num_iterations):
#         Y_hat = theta_1 * X + theta_0  # forward prop
#         distance = Y_hat - Y  # intermediate variable
#         cost = (distance ** 2).sum() / 2 / m  # cost function
#         history.append(cost)  # record cost history
#         d1, d0 = np.dot(distance, X) / m, distance.sum() / m  # gradient
#
#         # update parameter
#         theta_1 -= learning_rate * d1
#         theta_0 -= learning_rate * d0
#
#
# start = time.time()
# iterate(1_000_000, 0, 0)
# end = time.time()
# print(end - start)
