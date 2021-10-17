# %% import packages and data
import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt('./testSet.txt')
m, n = data.shape
X = np.concatenate((np.ones((m, 1)), data[:, :2]), axis=1)
Y = data[:, -1].reshape(m, 1)


# %% training the model
learning_rate = .1
theta = np.zeros((n, 1))
history = []
num_iterations = 10_000

for i in range(num_iterations):
    Y_hat = 1 / (1 + np.exp(-np.dot(X, theta)))
    cost = -(Y * np.log(Y_hat) +
             (1 - Y) * np.log(1 - Y_hat)).sum() / m
    history.append(cost)
    dJ = np.dot(X.T, Y_hat - Y) / m
    theta -= learning_rate * dJ

plt.plot(history)
plt.grid()
plt.title('Cost History')
plt.ylabel('Cost')
plt.xlabel('Iteration')
# plt.savefig('cost.png', dpi=300)
plt.show()

# %% plot the data and the decision boundary
x1 = np.meshgrid(np.arange(X[:, 1].min()-1, X[:, 1].max()+1))
x2 = np.meshgrid(np.arange(X[:, 2].min()-1, X[:, 2].max()+1))
xx1, xx2 = np.meshgrid(x1, x2)
yy = 1 / (1 - np.exp(
    - np.concatenate([np.ones((xx1.size, 1)),
                      xx1.reshape(-1, 1),
                      xx2.reshape(-1, 1)], axis=1) @ theta).reshape(xx1.shape))
plt.contourf(xx1, xx2, yy, cmap=plt.cm.Spectral, alpha=0.4)

for x1, x2, y in zip(X[:, 1], X[:, 2], Y):
    plt.scatter(x1, x2, c='C0' if y else 'C2')

# plt.savefig('boundary.png', dpi=300)
plt.show()
