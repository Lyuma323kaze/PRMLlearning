import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations
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
# gradient of square loss
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
              alpha = 0.05,
              para = None):
    gamma = ini_value
    for _ in range(max_iter):
        def f_plus_h(X):
            return f(X) + gamma * h(X, predict = True, para = para)
        grad = compute_gradient_square(X, y, f_plus_h) * h(X, predict = True, para = para)
        gamma -= alpha * grad.sum()
    return gamma

# determine main features, return 4 indices
def det_main_feature(X, y, number = 4, alpha = 1e-5, epsilon = 1e-1, C = 0.5, max_iter = 1000):
    w = lasso(X, y, alpha = alpha, epsilon = epsilon, C = C, max_iter = max_iter)
    w = np.abs(w)
    indices = np.argpartition(w, -number)[-number:]
    sorted_indices = indices[np.argsort(-w[indices])]
    return sorted_indices   # the 4 main features ordered by the values going down

# do gradient boosting
def gradient_boost(data_train, loss_tuple, iterations, weak_clasfers):
    m = iterations
    X = data_train[:, :-1]
    y = data_train[:, -1]
    # containers
    r_ls = []
    para = []
    h_ls = weak_clasfers
    gamma_ls = []
    # initialization
    f0 = lambda x: 0
    gradient = loss_tuple[1]
    f = f0
    # iteration
    for k in range(m):
        r = - gradient(X, y, f)
        r_ls.append(r)

        h = h_ls[k % 2]        # odd by tree, even by linear
        training_data = np.column_stack((X, r))
        paramter = h(data_train = training_data, compute = False)     # return para
        para.append(paramter)

        gamma = min_gamma(X, y, f, h, para = paramter)       # return gamma
        gamma_ls.append(gamma)

        def new_f(X):
            value = f0(X)
            paras = para.copy()
            gammas = gamma_ls.copy()
            for i in range(len(gammas)):
                h = h_ls[i % 2]
                value += gammas[i] * h(X_test=X, para=paras[i], predict=True)[2]
            return value

        f = new_f

    r_ls = np.array(r_ls)
    gamma_ls = np.array(gamma_ls)
    return f, r_ls, gamma_ls

def plot_main_features(X, y, main_features, f: callable, threshold = 0.5, name = None, m = None):
    feature_pairs = list(combinations(main_features, 2))
    fig, axes = plt.subplots(1, len(feature_pairs), figsize=(5 * len(feature_pairs), 5))
    e = threshold

    # for every feature combination
    for ax, (dim1, dim2) in zip(axes, feature_pairs):
        # generate mesh
        x_min, x_max = X[:, dim1].min() - e, X[:, dim1].max() + e
        y_min, y_max = X[:, dim2].min() - e, X[:, dim2].max() + e
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        # the feature matrix
        grid = np.zeros((xx.ravel().shape[0], X.shape[1]))
        grid[:, dim1] = xx.ravel()
        grid[:, dim2] = yy.ravel()

        # other dimensions as mean
        for d in range(X.shape[1]):
            if d not in [dim1, dim2]:
                grid[:, d] = np.mean(X[:, d])

        # get prediction
        Z = f(grid).reshape(xx.shape)

        # decision boundary
        ax.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')

        # 绘制样本点
        ax.scatter(X[:, dim1], X[:, dim2], c=y, edgecolors='k',
                   cmap='coolwarm', s=50)

        ax.set_xlabel(f'Feature {dim1}')
        ax.set_ylabel(f'Feature {dim2}')
        ax.set_title(f'Decision Boundary (Features {dim1} vs {dim2})')

    plt.tight_layout()
    plt.show()
    if name is not None:
        file_folder = name
        if not os.path.exists(file_folder):
            os.makedirs(file_folder)
        if not os.path.exists(os.path.join(file_folder, f'M = {m}')):
            os.makedirs(os.path.join(file_folder, f'M = {m}'))
        file_subfolder = os.path.join(file_folder, f'M = {m}')
        file_path = os.path.join(file_subfolder, f'plot_of_{name}@M = {m}.png')
        plt.savefig(file_path)
        plt.close()


# import data from txt
data_ini = np.genfromtxt('breast-cancer-wisconsin.txt', missing_values = '?', filling_values = np.nan)
data = data_ini[~np.isnan(data_ini).any(axis = 1)]
data = data[:, 1:]  # to be refined for the WDBC set

# import data from csv (to be done)


# training set and test set
data_train, data_test = train_test_split(data, train_size=0.8, test_size=0.2)

# loss function and gradient tuple
loss_sqr = (square, compute_gradient_square)

# weak classifiers
h_ls = [linear, tree]

# iteration number M
m = 5

if __name__ == '__main__':
    f, r_ls, gamma_ls = gradient_boost(data_train, loss_sqr, m, h_ls)
    # print(f(data_train))
    acc = compute_accuracy(data_test[:, :-1], data_test[:, -1], f)
    print(f'accuracy = {acc}')
    X, y = data[:, :-1], data[:, -1]
    main_feature_indices = det_main_feature(X, y, number = 3)   # 3 main indices
    # plot_main_features(X, y, main_feature_indices, f = f, threshold = 0.5)     # plot decision boundary


