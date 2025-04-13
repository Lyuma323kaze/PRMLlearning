from sklearn.datasets import make_swiss_roll as make_roll
from sklearn.datasets import make_s_curve as make_curve
import numpy as np
import matplotlib.pyplot as plt

# Swiss roll
X, color = make_roll(n_samples=1000, noise=0.1, random_state = 123)
Y, colour = make_curve(n_samples=1000, noise=0.1, random_state = 42)


def compute_pca(X, color, k, name = None):
    n = X.shape[0]
    sigma = (X.T @ X) / n
    eig_values, eig_vectors = np.linalg.eig(sigma)
    for i in range(eig_vectors.shape[1]):
        eig_vectors[:, i] = eig_vectors[:, -i - 1]
    U = eig_vectors[:, :k]
    Z = X @ U

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c = color, marker='o', cmap = 'coolwarm')
    plt.colorbar(scatter, label = 'color values')
    plt.title(f'Embedded data by PCA for {name}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'PCA for {name}.png')
    plt.close()
    return Z, U


def compute_lle(X, color, dim, k, a_w = 0.1, a_z = 0.1, e_w = 1e-2, e_z = 1e-4, max_iter_w = 1000, max_iter_z = 1000, name = None):
    # initialize W
    W = np.ones([X.shape[0], k]) / k

    # determine the neighbors
    def determine_neighbor(X, k):
        X_sq = np.sum(X ** 2, axis = 1)
        D = X_sq[:, None] - 2 * np.dot(X, X.T) + X_sq[None, :]
        np.fill_diagonal(D, np.inf)
        indices = np.argsort(D, axis = 1)[:, :k]
        return indices
    indices = determine_neighbor(X, k)

    def compute_gradient_WT(X, W):
        S = X[indices]
        weight_sum = np.einsum('ik,ikd->id', W, S)
        diff = X - weight_sum
        return -2 * np.einsum('id, ikd->ik', diff, S)

    def determine_w(X, W, epsilon, alpha, max_iter = max_iter_w):
        a = alpha
        e = epsilon
        for _ in range(max_iter):
            W_old = W.copy()
            grad = compute_gradient_WT(X, W)
            W -= a * grad
            W /= W.sum(axis = 1, keepdims = True)
            if np.linalg.norm(W - W_old) < e:
                break
        return W

    def compute_gradient_z(W, Z):
        S = Z[indices]
        weight_sum = np.einsum('ik,ikd->id', W, S)
        diff = Z - weight_sum
        return 2 * diff

    def determine_z(W, dim, epsilon, alpha, max_iter = max_iter_z):
        a_0 = alpha
        e = epsilon
        Z = np.ones([X.shape[0], dim])
        for j in range(max_iter):
            Z_old = Z.copy()
            grad = compute_gradient_z(W, Z)
            if j == 0:
                a = 1e15    # The initial gradient is always too small that Z cannot be calculated correctly
            else:
                a = a_0     # After initialization, remain the raw learning rate
            Z -= a * grad
            if (np.linalg.norm(Z - Z_old)) < e:
                break
        return Z
    W_deter = determine_w(X, W, e_w, a_w)
    Z_deter = determine_z(W_deter, dim, e_z, a_z)
    print(f'W_deter: {W_deter}')
    print(W_deter.shape)
    print(f'Z_deter: {Z_deter}')
    print(Z_deter.shape)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(Z_deter[:, 0], Z_deter[:, 1], c=color, marker='o', cmap='coolwarm')
    plt.colorbar(scatter, label='color values')
    plt.title(f'Embedded data by LLE at k = {k} for {name}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'LLE for {name}.png')
    plt.show()
    return

alpha = 1e-3    # learning rate
epsilon = 1e-1  # tolerance
max_iter = 5000 # maximum iteration

# the learning rate in determining z is specified

# compute Swiss roll case
compute_pca(X, color, 2, name = 'roll')
compute_lle(X, color, 2, 25, a_w = alpha, a_z = 1e-2, max_iter_w = max_iter, max_iter_z = max_iter, name = 'roll')

# compute S curve case
compute_pca(Y, colour, 2, name = 'S_curve')
compute_lle(Y, colour, 2, 35, a_w = alpha, a_z = 1e-2, max_iter_w = max_iter, max_iter_z = max_iter, name = 'S_curve')