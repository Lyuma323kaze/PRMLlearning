import numpy as np
import matplotlib.pyplot as plt


def grad_L(x):
    if x.shape[0] != 2:
        raise ValueError('x must be a 2-dimensional array')
    grad = 2 * np.array((x[0], 99 * x[1]))
    return grad

def loss(x):
    if x.shape[0] != 2:
        raise ValueError('x must be a 2-dimensional array')
    return x[0] ** 2 + 99 * x[1] ** 2

def gradient_descent(x, eta, iterations, m = 0.):
    diff = np.ones_like(x)
    x_ls = []
    loss_ls = []
    x_ls.append(x)
    loss_ls.append(loss(x))
    value = x.copy()
    for _ in range(iterations):
        grad = grad_L(value)
        diff = - eta * grad + m * value
        value = value + diff
        x_ls.append(value)
        loss_ls.append(loss(value))
    x_ls = np.array(x_ls)
    return x_ls, loss_ls

eta = 0.02
m = 0.9
iter = 100

np.random.seed(941)
x0 = np.random.uniform(low = -0.1, high = 0.1, size = 2)
print(x0)

x_simple, loss_simple = gradient_descent(x0, eta, iter)
x_momentum, loss_momentum = gradient_descent(x0, eta, iter, m)

plt.plot(x_simple[:, 0], x_simple[:, 1])
plt.title('x_simple')
plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig('x_simple.png')

plt.plot(x_momentum[:, 0], x_momentum[:, 1])
plt.title('x_momentum')
plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig('x_momentum.png')

plt.plot(loss_simple)
plt.title('loss_simple')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.savefig('loss_simple.png')

plt.plot(loss_momentum)
plt.title('loss_momentum')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.savefig('loss_momentum.png')
