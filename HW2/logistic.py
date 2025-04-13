import numpy as np
import matplotlib.pyplot as plt

data_ini = np.genfromtxt('breast-cancer-wisconsin.txt', missing_values = '?', filling_values = np.nan)
data = data_ini[~np.isnan(data_ini).any(axis = 1)]
threshold = 0

def ones(X):
    return np.ones(X.shape[0]).reshape(-1, 1)

def code(set):
    cd = set[:, 0]
    return cd

def label(set):
    lb = set[:, -1]
    return lb

def chara(set):
    chra = set[:, 1:-1]
    return chra

def sample_1(set):
    mask_1 = (set[:, -1] == 1)
    samp = set[mask_1]
    return samp

def sample_0(set):
    mask_0 = (set[:, -1] == 0)
    samp = set[mask_0]
    return samp

def case_train(sample, case_num):
    return sample[:case_num]

# return test set with all labels
def case_test(sample, case_num):
    i = 1
    test_case = []
    while i <= case_num:
        test_case.append(sample[-i])
        i += 1
    test_case = np.array(test_case)
    return test_case

def logistic(dat, p, critilos):
    X = chara(dat)
    y = label(dat)
    b = 0
    w = np.random.randn(X.shape[1],1)
    los = 10000
    i = 0
    losses = []
    accies = []
    while los >= critilos:
        los = loss(X, y, w, b)
        gw = gradw(X, y, w, b) / len(X)
        gb = gradb(X, y, w, b) / len(X)
        w -= p * gw.T
        b -= p * gb
        i += 1
        threshold = 0.5
        projection = sigmoid(X @ w + (b * ones(X @ w)))
        prediction = (projection > threshold).astype(int)
        accy = np.mean(prediction == y)
        accies.append(accy)
        losses.append(los)
        if i >= 5000:
            break
    return w, b, losses, accies

def gradw(X, y, w, b):
    s = np.zeros((1, X.shape[1]))
    for i in range(0, X.shape[0]):
        c = X[i,:] @ w + b
        c = c[0]
        s += (sigmoid(c) / (1 + np.exp(c))) * (sigmoid(c) - y[i]) * X[i, :]
        # print(np.shape(X[i,:]))
    return s

def gradb(X, y, w, b):
    s = 0
    for i in range(0, X.shape[0]):
        c = (X[i,:] @ w + b)
        c = c[0]
        s += (sigmoid(c) / (1 + np.exp(c))) * (sigmoid(c) - y[i])
    return s

def loss(X, y, w, b):
    t = 0.5 * np.sum((sigmoid(X @ w + b * ones(X @ w)) - y.reshape(-1, 1)) ** 2)
    return t

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def accuracy(dat, p, cri):
    X = chara(dat)
    y = label(dat)
    w, b, losses, accies = logistic(dat, p, cri)
    w_norm = w / np.linalg.norm(w)
    print(w_norm)
    # X0 = sample_0(dat)
    # X1 = sample_1(dat)
    # threshold = (np.mean(sigmoid(chara(X0) @ w + b * ones(X0))) + np.mean(sigmoid(chara(X1) @ w + b * ones(X1)))) / 2
    plt.figure(figsize=(10,5))
    plt.plot(losses, label = 'Loss', color = 'red')
    # plt.plot(accies, label = 'Accuracy', color = 'blue')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Loss vs. Epoch')
    plt.show()
    final_accuracy = accies[-1]
    return final_accuracy

print('final accuracy:', accuracy(data, 0.1, 0.01))





