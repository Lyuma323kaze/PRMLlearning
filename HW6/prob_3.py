import os
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as kms
from sklearn.mixture import GaussianMixture as gmm
from scipy.stats import mode
from sklearn.decomposition import PCA

data_tot = np.load('toy_mnist_2024.npz')
X = data_tot['X'].astype('float32') / 255
y = data_tot['y'].astype('float32')

def train_test_split(X, y, train_size=None, test_size=None):
    n_samples = X.shape[0]
    test_num = int(test_size)
    train_num = int(train_size)
    # 生成随机索引并分割
    indices = np.random.permutation(n_samples)
    test_indices = indices[:test_num]
    train_indices = indices[test_num: test_num + train_num]

    # 分离数据
    X_train = X[train_indices,]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

def det_elbow(max_clusters, X_tr):
    je_ls = []
    clusters = np.arange(2, max_clusters + 1, 1)
    for i in range(2, max_clusters + 1):
        model_kms = kms(
            n_clusters=i,
            init='k-means++',
            n_init=50,
            max_iter=500,
            tol=1e-4,
            random_state=None
        )
        X_reshaped = X_tr.reshape(X_tr.shape[0], -1)
        pca = PCA(n_components=0.95)  # 保留 95% 方差
        X_pca = pca.fit_transform(X_reshaped)
        model_kms.fit(X_pca)
        je_ls.append(model_kms.inertia_)
    je_ls = np.array(je_ls)
    je_ls.reshape((-1, 1))
    plt.figure(figsize=(8, 4))
    plt.plot(clusters, je_ls,
            marker='o',
            linestyle='-',
            linewidth=1,
            markersize=4,
            color='red',
            label='Err1')
    # 添加标签和标题
    plt.title("$J_e$ vs Clusters", fontsize=12)
    plt.xlabel("Clusters", fontsize=10)
    plt.ylabel("$J_e$", fontsize=10)

    # 添加网格和图例
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # 保存和显示
    plt.tight_layout()  # 自动调整布局
    plt.show()
    return 0

def plt_means(clusters, X_tr):
    model_kms = kms(
        n_clusters=clusters,
        init='k-means++',
        n_init=50,
        max_iter=300,
        tol=1e-4,
        random_state=None
    )
    X_reshaped = X_tr.reshape(X_tr.shape[0], -1)
    model_kms.fit(X_reshaped)
    mean_ls = model_kms.cluster_centers_.reshape(clusters, 28, 28)
    for k in range(len(mean_ls)):
        mean = mean_ls[k]
        plt.imshow(mean, cmap='gray_r', vmin=0, vmax=1)  # 使用灰度颜色映射
        plt.axis('off')  # 可选：隐藏坐标轴
        plt.savefig(f'mean_KMeans_elbow={elbow_cluster}_{k}.png', bbox_inches='tight', pad_inches=0)
        plt.show()
    return 0

def det_accuracy(clusters, X_tr, X_te, y_te):
    model_kms = kms(
        n_clusters=clusters,
        init='k-means++',
        n_init=50,
        max_iter=600,
        tol=1e-4,
        random_state=None
    )
    X_reshaped = X_tr.reshape(X_tr.shape[0], -1)
    model_kms.fit(X_reshaped)
    X_te_reshaped = X_te.reshape(X_te.shape[0], -1)

    pred = model_kms.predict(X_te_reshaped)
    pred_one_hot = np.eye(10, dtype = np.float32)[pred]

    cluster_to_majority_label = {}
    y_te_flatten = np.argmax(y_te, axis=1)
    for cluster in range(model_kms.n_clusters):
        mask = (pred == cluster)
        labels_in_cluster = y_te_flatten[mask]
        if len(labels_in_cluster) > 0:
            majority_label = mode(labels_in_cluster).mode  # 取众数
            cluster_to_majority_label[cluster] = majority_label
    predicted_labels = np.array([cluster_to_majority_label[c] for c in pred])

    mask = np.array((predicted_labels == y_te_flatten))
    acrcy = np.sum(mask) / mask.shape[0]
    return acrcy

def EM_mnist(X, n_clusters):
    X_reshaped = X.reshape(X.shape[0], -1)
    model_gmm = gmm(n_components=n_clusters, covariance_type='full', random_state=42)
    model_gmm.fit(X_reshaped)
    centers = model_gmm.means_
    for k in range(len(centers)):
        mean = centers[k]
        mean_reshape = mean.reshape(28, 28)
        plt.imshow(mean_reshape, cmap='gray_r', vmin=0, vmax=1)  # 使用灰度颜色映射
        plt.axis('off')  # 可选：隐藏坐标轴
        plt.savefig(f'mean_EM_{k}.png', bbox_inches='tight', pad_inches=0)
        plt.show()
    return centers

def det_accr_EM(clusters, X_tr, X_te, y_te):
    X_reshaped = X_tr.reshape(X_tr.shape[0], -1)
    model_gmm = gmm(n_components=clusters, covariance_type='full', random_state=42)
    model_gmm.fit(X_reshaped)
    X_te_reshaped = X_te.reshape(X_te.shape[0], -1)

    pred = model_gmm.predict(X_te_reshaped)

    cluster_to_majority_label = {}
    y_te_flatten = np.argmax(y_te, axis=1)
    for cluster in range(clusters):
        mask = (pred == cluster)
        labels_in_cluster = y_te_flatten[mask]
        if len(labels_in_cluster) > 0:
            majority_label = mode(labels_in_cluster).mode  # 取众数
            cluster_to_majority_label[cluster] = majority_label
    predicted_labels = np.array([cluster_to_majority_label[c] for c in pred])

    mask = np.array((predicted_labels == y_te_flatten))
    acrcy = np.sum(mask) / mask.shape[0]
    return acrcy


X_train, X_test, y_train, y_test = train_test_split(X, y, 800, 200)
det_elbow(70, X_train)

elbow_cluster = 10

plt_means(elbow_cluster, X_train)

accry = det_accuracy(elbow_cluster, X_train, X_test, y_test)
print(f'accuracy: {accry}')

EM_mnist(X, elbow_cluster)

accry_EM = det_accr_EM(elbow_cluster, X_train, X_test, y_test)
print(f'accuracy EM: {accry_EM}')
