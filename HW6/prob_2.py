import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

# real parameters
mean1 = np.array([0, 0])
mean2 = np.array([2, 2])
cov0 = [[1, 0], [0, 1]]
pi0 = [2/3 , 1 - 2/3]

nor1 = mvn(mean1, cov0)
nor2 = mvn(mean2, cov0)
nor = [nor1, nor2]

# sample number
N = 3000

x_range = np.arange(-3, 5.01, 0.01)
y_range = np.arange(-3, 5.01, 0.01)


def gene_samples(N, pi):
    z = np.random.choice(len(pi), size = N, p = pi)
    data = np.zeros((N, 2))
    for i in range(len(pi)):
        mask = (z == i)
        data[mask] = nor[i].rvs(np.sum(mask))
    return data

data = gene_samples(N, pi0)

# initialized parameters


def init_data(N):
    pi = np.array([0.5, 0.5])
    mu = np.array([[-1, -1], [3.3, 3.3]])
    cov = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    gamma = np.zeros((N, 2))
    return pi, mu, cov, gamma
pi, mu, cov, gamma = init_data(N)
print(f'init pi = {pi}')
print(f'init mu = {mu}')
print(f'init cov = {cov}')
print(f'init gamma = {gamma}')

def EM(data, pi1, mu1, cov1, gamma1, max_iter):
    k = 0
    err1 = []
    err2 = []
    pi = np.copy(pi1).reshape(1, -1)
    mu = np.copy(mu1)
    cov = np.copy(cov1).astype(np.float64)
    gamma = np.copy(gamma1)
    while k < max_iter:
        for i in range(2):
            gamma[:, i] = pi[0, i] * mvn.pdf(data, mean = mu[i], cov = cov[i], allow_singular = True)
        gamma /= gamma.sum(axis = 1, keepdims = True)
        Nk = gamma.sum(axis = 0, keepdims = True)
        for i in range(2):
            mu[i] = ((data * gamma[:, i, np.newaxis]).sum(axis = 0)) / Nk[0, i]
            diff = data - mu[i]
            cov[i] = (gamma[:, i, np.newaxis, np.newaxis] * ((diff[:, :, np.newaxis]
                        * diff[:, np.newaxis, :]))).sum(axis = 0) / Nk[0, i]
            cov[i] = 0.5 * (cov[i] + cov[i].T)  # 强制对称
            cov[i] += np.eye(2) * 1e-6
        pi = Nk / len(data)
        err1.append(np.linalg.norm(mu[0] - mean1))
        err2.append(np.linalg.norm(mu[1] - mean2))
        k += 1
    return pi, mu, cov, (err1, err2)

pi_ob, mu_ob, cov_ob, err_ob = EM(data, pi, mu, cov, gamma, 1000)
print(f'estimated mu = {mu_ob}')
print(f'estimated cov = {cov_ob}')
print(f'estimated pi = {pi_ob}')

def plt_agst_N(min_iter, max_iter, samp_iters = 1000):
    idct = np.arange(min_iter, max_iter + 1, 10)
    errls = []
    for Nt in idct:
        datat = gene_samples(Nt, pi0)
        pit, mut, covt, gammat = init_data(Nt)
        pie, mue, cove, erre = EM(datat, pit, mut, covt, gammat, samp_iters)
        err_use = (erre[0][-1], erre[1][-1])
        errls.append(err_use)

    errls = np.array(errls)
    print(errls[:, 1].shape)
    print(idct.shape)
    plt.figure(figsize=(8, 4))
    plt.plot(idct, errls[:, 0],
            marker='o',
            linestyle='-',
            linewidth=1,
            markersize=4,
            color='red',
            label='Err1')
    plt.plot(idct, errls[:, 1],
            marker='o',
            linestyle='-',
            linewidth=1,
            markersize=4,
            color='blue',
            label='Err2')

    # 添加标签和标题
    plt.title("Error of Mean vs Sample Number", fontsize=12)
    plt.xlabel("Sample Number", fontsize=10)
    plt.ylabel("Error", fontsize=10)

    # 添加网格和图例
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # 保存和显示
    plt.tight_layout()  # 自动调整布局
    plt.show()
    return 0

def plt_agst_t(min_iter, max_iter, samp_number = 2000):
    idct = np.arange(min_iter, max_iter + 1, 5)
    errls = []
    datat = gene_samples(samp_number, pi0)
    for step in idct:
        pit, mut, covt, gammat = init_data(samp_number)
        pie, mue, cove, erre = EM(datat, pit, mut, covt, gammat, step)
        err_use = (erre[0][-1], erre[1][-1])
        errls.append(err_use)

    errls = np.array(errls)
    print(errls[:, 1].shape)
    print(idct.shape)
    plt.figure(figsize=(8, 4))
    plt.plot(idct, errls[:, 0],
             marker='o',
             linestyle='-',
             linewidth=1,
             markersize=4,
             color='red',
             label='Err1')
    plt.plot(idct, errls[:, 1],
             marker='o',
             linestyle='-',
             linewidth=1,
             markersize=4,
             color='blue',
             label='Err2')

    # 添加标签和标题
    plt.title("Error of Mean vs Steps", fontsize=12)
    plt.xlabel("Step", fontsize=10)
    plt.ylabel("Error", fontsize=10)

    # 添加网格和图例
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # 保存和显示
    plt.tight_layout()  # 自动调整布局
    plt.show()
    return 0

plt_agst_N(10, 3000)
plt_agst_t(10, 500)


