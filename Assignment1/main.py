import numpy as np
import matplotlib.pyplot as plt

from weak_classifiers import linear_clsfer_fisher as linear
from weak_classifiers import tree_clsfer as tree
from LASSO import lasso_regression as lasso

# splitting
def train_test_split(data, train_size=None, test_size=None):
    n_samples = data.shape[0]
    test_num = int(test_size * n_samples)
    train_num = int(train_size * n_samples)
    # 生成随机索引并分割
    indices = np.random.permutation(n_samples)
    test_indices = indices[:test_num]
    train_indices = indices[test_num: test_num + train_num]

    # 分离数据
    data_train = data[train_indices]
    data_test = data[test_indices]

    return data_train, data_test

# cross entropy (not used)
def cross_entropy(X, y, f: callable):
    def compute_gradient_entropy():
        return
    return

# square loss
def square(X, y, f: callable):
    return (y - f(X)) ** 2

def compute_gradient_square(X, y, f: callable):
    return 2 * (y - f(X))

# return accuracy
def compute_accuracy(X, y, f: callable):
    mask = np.array(f(X) == y)
    accr = mask.astype(int).mean()
    return accr

# return gamma
def min_gamma(X, y, f: callable,
              h: callable,
              ini_value = 0.1,
              max_iter = 50,
              alpha = 0.05):
    gamma = ini_value
    for _ in range(max_iter):
        def f_plus_h(X):
            return f(X) + h(X)
        grad = compute_gradient_square(X, y, f_plus_h) * h(X)
        gamma -= alpha * grad.sum()
    return gamma

# determine main features
def det_main_feature(X, y, alpha = 1e-2, epsilon = 1e-2, C = 0.1, max_iter = 1000):
    w = lasso(X, y, alpha = alpha, epsilon = epsilon, C = C, max_iter = max_iter)
    w = np.abs(w)
    indices = np.partition(w, -4)[-4:]
    sorted_indices = indices[np.argsort(-w[indices])]
    return sorted_indices   # the 4 main features ordered by the values going down

# do gradient boosting
def gradient_boost(data_train, loss_tuple, iterations):
    m = iterations
    X = data_train[:, :-1]
    y = data_train[:, -1]
    # containers
    r_ls = []
    para = []
    h_ls = [linear, tree]
    gamma_ls = []
    # initialization
    f = lambda x: 0
    loss = loss_tuple[0]
    gradient = loss_tuple[1]
    # iteration
    for k in range(m):
        r = - gradient(X, y, f)
        r_ls.append(r)

        h = h_ls[k // 2]        # odd by tree, even by linear
        paramter = h(X, r, compute = False)     # return para
        para.append(paramter)

        gamma = min_gamma(X, y, f, h)       # return gamma
        gamma_ls.append(gamma)

        new_f = lambda X, param = paramter: f(X) + gamma * h(X, para = param, predict = True)
        f = new_f
    r_ls = np.array(r_ls)
    gamma_ls = np.array(gamma_ls)
    return f, r_ls, gamma_ls

# import data
data_ini = np.genfromtxt('breast-cancer-wisconsin.txt', missing_values = '?', filling_values = np.nan)
data = data_ini[~np.isnan(data_ini).any(axis = 1)]

# training set and test set
data_train, data_test = train_test_split(data, train_size=0.8, test_size=0.2)

# loss function and gradient tuple
loss_sqr = (square, compute_gradient_square)


if __name__ == '__main__':
    f, r_ls, gamma_ls = gradient_boost(data_train, loss_sqr, 100)[0]
