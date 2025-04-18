import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations
from matplotlib.lines import Line2D
from weak_classifiers import linear_fisher_gaussian_kclass as linear
from weak_classifiers import tree_clsfer as tree
from sklearn.linear_model import Lasso
import pandas as pd

np.random.seed(114)

# splitting
def train_test_split(_data, train_size=None, test_size=None):
    n_samples = _data.shape[0]
    test_num = int(test_size * n_samples)
    train_num = int(train_size * n_samples)
    # 生成随机索引并分割
    indices = np.random.permutation(n_samples)
    test_indices = indices[:test_num]
    train_indices = indices[test_num: test_num + train_num]

    # 分离数据
    data_train = _data[train_indices]
    data_test = _data[test_indices]

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
    return -2 * (y - f(X))

# return accuracy
def compute_accuracy(X, y, f: callable, threshold = 0.5):
    difference = np.abs(y - f(X))
    mask = (difference <= threshold) | (f(X) > (np.max(y) + threshold))
    accr = mask.astype(int).mean()
    return accr

# return gamma
def min_gamma(X, y, f: callable,
              h: callable,
              ini_value = 1e-3,
              max_iter = 50,
              alpha = 0.05,
              para = None):
    gamma = ini_value
    for _ in range(max_iter):
        def f_plus_h(x):
            return f(x) + gamma * h(X_test = x, predict = True, para = para)
        grad = compute_gradient_square(X, y, f_plus_h) * h(X_test = X, predict = True, para = para)
        gamma -= alpha * grad.mean()
    return gamma

# determine main features, return 4 indices
def det_main_feature(X, y, number = 4, alpha = 1e-5, epsilon = 1e-1, C = 0.5, max_iter = 1000):
    lasso_obj = Lasso(alpha = 0.1)
    lasso_obj.fit(X, y)
    w = lasso_obj.coef_
    w = np.abs(w)
    indices = np.argpartition(w, -number)[-number:]
    sorted_indices = indices[np.argsort(-w[indices])]
    return sorted_indices   # the 4 main features ordered by the values going down

# do gradient boosting
def gradient_boost(_data_train, loss_tuple, iterations, weak_clasfers):
    m = iterations
    X_train = _data_train[:, :-1]
    y_train = _data_train[:, -1]
    # containers
    r_ls = []
    para = []
    h_ls = weak_clasfers
    gamma_ls = []
    # initialization
    f0 = lambda x: np.zeros(len(x))
    gradient = loss_tuple[1]
    f = f0
    y_trainn = y_train.copy()
    # iteration
    for k in range(m):
        r = - gradient(X_train, y_train, f)
        r_ls.append(r)

        h = h_ls[k % 2]        # odd by tree, even by linear
        training_data = np.column_stack((X_train, r))
        paramter = h(_data_train = training_data, compute = False)     # return para
        para.append(paramter)

        gamma = min_gamma(X_train, y_trainn, f, h, para = paramter)  # return gamma
        gamma_ls.append(gamma)

        def new_f(X):
            value = 0
            paras = para.copy()
            gammas = gamma_ls.copy()
            for i in range(len(gammas)):
                h = h_ls[i % 2]
                delta_value = gammas[i] * h(X_test=X, para=paras[i], predict=True)
                value += delta_value
            return value

        f = new_f
    r_ls = np.array(r_ls)
    gamma_ls = np.array(gamma_ls)
    return f, r_ls, gamma_ls

def plot_main_features(X, main_features,
                       f: callable,
                       threshold = 0.5,
                       name = None,
                       m = None):
    feature_pairs = list(combinations(main_features, 2))
    e = threshold

    valid_pairs = []
    precomputed = []  # save(xx, yy, Z)

    # preprocessing
    for dim1, dim2 in feature_pairs:
        # mesh
        x_min, x_max = X[:, dim1].min() - e, X[:, dim1].max() + e
        y_min, y_max = X[:, dim2].min() - e, X[:, dim2].max() + e
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        # specific matrx
        grid = np.zeros((xx.ravel().shape[0], X.shape[1]))
        grid[:, dim1] = xx.ravel()
        grid[:, dim2] = yy.ravel()

        # other dim with mean
        for d in range(X.shape[1]):
            if d not in [dim1, dim2]:
                grid[:, d] = np.mean(X[:, d])

        # prediction
        Z = f(grid).reshape(xx.shape)
        if Z.var() < 1e-10:
            continue  # skip small variance result

        valid_pairs.append((dim1, dim2))
        precomputed.append((xx, yy, Z))

    # no effective pairs
    if not valid_pairs:
        print("No valid feature pairs to plot.")
        return

    # generate subplots by effective pairs
    n_valid = len(valid_pairs)
    fig, axes = plt.subplots(1, n_valid, figsize=(5 * n_valid, 5))
    if n_valid == 1:
        axes = [axes]  # axes as list

    # obj for generate colorbar
    contourf_obj = None

    # plot effective pairs
    for ax, (dim1, dim2), (xx, yy, Z) in zip(axes, valid_pairs, precomputed):
        contorf = ax.contourf(xx, yy, Z, alpha=1, cmap='coolwarm', vmin = -0.2, vmax = 0.7, levels = 100)
        if contourf_obj is None:
            contourf_obj = contorf
        contour = ax.contour(xx, yy, Z, levels=[threshold],
                   colors='k', linewidths=4, linestyles='solid')
        legend_lines = [
            Line2D([0], [0], color='k', linewidth=4, linestyle='solid', label='f(X) = 0.5')  # 决策边界
        ]
        ax.legend(handles=legend_lines, loc='upper left', fontsize=10, frameon=True)
        ax.set_xlabel(f'Feature {dim1}')
        ax.set_ylabel(f'Feature {dim2}')
        ax.set_title(f'Decision Boundary on Features {dim1} vs {dim2}, m = {m}')

    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes((0.92, 0.15, 0.015, 0.7))  # position of colorbar
    z_values = contourf_obj.get_array()
    z_min = z_values.min()
    z_max = z_values.max()
    cbar = fig.colorbar(contorf, cax=cbar_ax, label='Prediction Score')
    cbar.set_ticks(np.linspace(z_min, z_max, 5))

    if name is not None:
        file_folder = name
        if not os.path.exists(file_folder):
            os.makedirs(file_folder)
        file_path = os.path.join(file_folder, f'{name}@M = {m}.png')
        plt.savefig(file_path)
        plt.close()
    plt.show()
    return

def plot_gamma(gamma_ls, name = None):
    l = len(gamma_ls)
    if name == 'tree':
        x = np.linspace(1, 2*l-1, l)
    if name == 'linear':
        x = np.linspace(2, 2*l, l)
    plt.plot(x, gamma_ls, marker='o')
    plt.title(fr'Change of $\gamma_m$ for {name} classifier')
    plt.xticks(ticks=x,
               labels=x.astype(str),
               ha='right',
               fontsize=10)
    plt.xlabel('m')
    plt.ylabel(r'$\gamma$')
    if name is not None:
        file_folder = 'gamma'
        if not os.path.exists(file_folder):
            os.makedirs(file_folder)
        file_path = os.path.join(file_folder, f'gamma@{name}.png')
        plt.savefig(file_path)
    plt.show()
    return

def judge_r_var(r_ls):
    diff = np.array([r_ls[i+1] - r_ls[i] for i in range(len(r_ls) - 1)])
    signed_diff = np.sign(diff)
    has_positive = np.any(diff > 0, axis=1)
    has_negative = np.any(diff < 0, axis=1)
    mono = ~(has_positive & has_negative)
    mono = np.array(mono).astype(int)
    mono_rate = mono.mean()
    print(f'monotonical rate: {mono_rate}')

    signed_diff = signed_diff.mean(axis=1)
    signed_diff = np.sign(signed_diff)

    x = np.arange(2, len(r_ls) + 1)
    plt.plot(x, signed_diff, marker='o')
    plt.title(r'sgn($\overline{r_m - r_{m-1}}$) of $r_m$')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    plt.xlabel('m')
    plt.ylabel(r'$r_m$')
    file_folder = 'r_m'
    if not os.path.exists(file_folder):
        os.makedirs(file_folder)
    file_path = os.path.join(file_folder, f'r_m.png')
    plt.savefig(file_path)
    plt.show()
    return


# import data from csv
df = pd.read_csv("WDBC.csv")
data = df.iloc[:, 2:-1].values
labels = df["diagnosis"]
labels = labels.map({'M': 1, 'B': 0}).values
labels = labels.reshape(-1, 1)
data = np.hstack([data, labels])

# import data from csv (to be done)


# training set and test set
data_train, data_test = train_test_split(data, train_size=0.7, test_size=0.3)

# loss function and gradient tuple
loss_sqr = (square, compute_gradient_square)

# weak classifiers
h_ls = [tree, linear]

# iteration number M
m_ls = np.arange(1, 11)
m_fin = 20

# classification threshold
threshold = 0.5
name = 'Decision_Boundary'

# control parameters
plotting = False
analyzing = True

if __name__ == '__main__':
    if plotting:
        for m in m_ls:
            f, r_ls, gamma_ls = gradient_boost(data_train, loss_sqr, m, h_ls)
            acc = compute_accuracy(data_test[:, :-1], data_test[:, -1], f, threshold = threshold)
            print(f'm = {m}, accuracy = {acc}')
            X, y = data[:, :-1], data[:, -1]
            main_feature_indices = det_main_feature(X, y, number = 4)   # 3 main indices
            plot_main_features(data_test[:, :-1], main_feature_indices, f = f, threshold = 0.5, m = m, name = name)     # plot decision boundary

    if analyzing:
        f, r_ls, gamma_ls = gradient_boost(data_train, loss_sqr, m_fin, h_ls)
        gamma_tree = gamma_ls[[2*i for i in range(m_fin // 2)]]
        gamma_linear = gamma_ls[[2 * i + 1 for i in range(m_fin // 2)]]
        # plot_gamma(gamma_tree, name = 'tree')
        # plot_gamma(gamma_linear, name = 'linear')
        judge_r_var(r_ls)




