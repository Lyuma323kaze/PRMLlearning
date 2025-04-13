import numpy as np

data_X = [
    [-1.784, -0.192, 0.505, 0.346],
    [-0.749, 0.762, -1.823, 0.215],
    [-0.636, 0.996, 0.395, 0.747],
    [-0.637, -0.006, -0.051, 0.655],
    [-0.455, -0.811, -0.991, -0.268],
    [0.270, -0.633, -0.779, -1.255],
    [0.906, -1.307, -0.524, 2.032],
    [0.844, 1.084, -1.197, -0.255],
    [0.705, -1.287, 0.078, 0.984],
    [0.637, -0.378, 0.070, 1.140]
]
data_X = np.array(data_X)

data_y = [
    -1.077, -1.308, -1.248, -0.534, 0.868, 0.721, 1.534, -0.392, 1.443, -0.205
]
data_y = np.array(data_y)
# data_y.reshape(-1, 1)
max_iter = int(5e3)
alpha = 1e-2
epsilon = 1e-2

# numerically, w is a column vector
def compute_gradient_SSE(data_X, data_y, w):
    return 2 * (w.T @ data_X.T - data_y.T) @ data_X

def linear_regression(data_X, data_y, alpha, epsilon, max_iter = 1000):
    a = alpha # learning rate
    e = epsilon
    # initialize w
    w = np.ones(4)
    grad = compute_gradient_SSE(data_X, data_y, w)
    for _ in range(max_iter):
        w_old = w.copy()
        w -= a * grad
        w = w / np.linalg.norm(w)
        grad = compute_gradient_SSE(data_X, data_y, w)
        if np.linalg.norm(w - w_old) < e:
            break
    w = w / np.linalg.norm(w)
    return w

def compute_gradient_norm1(C, w):
    mask1 = np.array(w > 0).astype(float)
    mask0 = np.array(w == 0).astype(float)
    mask11 = np.array(w < 0).astype(float)
    grad = C * (mask1 + -1 * mask11)
    grad = grad.flatten()
    return grad


def lasso_regression(data_X, data_y, alpha, epsilon, C, max_iter = 1000):
    a = alpha
    e = epsilon
    # initialize w
    w = np.ones(4)
    grad = compute_gradient_SSE(data_X, data_y, w) + compute_gradient_norm1(C, w)
    for _ in range(max_iter):
        w_old = w.copy()
        w -= a * grad
        grad = compute_gradient_SSE(data_X, data_y, w) + compute_gradient_norm1(C, w)
        if np.linalg.norm(w - w_old) < e:
            break
    # w = w / np.linalg.norm(w)
    return w

w_linear = linear_regression(data_X, data_y, alpha, epsilon, max_iter)
w_lasso = lasso_regression(data_X, data_y, alpha, epsilon, 0.1, max_iter)

np.set_printoptions(suppress= True, precision = 4)
print(f'w_linear = {w_linear}')
print(f'w_lasso = {w_lasso}')

