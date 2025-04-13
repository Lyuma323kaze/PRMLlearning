import numpy as np
import matplotlib.pyplot as plt

data_ini = np.genfromtxt('breast-cancer-wisconsin.txt', missing_values = '?', filling_values = np.nan)
data = data_ini[~np.isnan(data_ini).any(axis = 1)]
threshold = 0
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
    i = 0
    train_case = []
    while i < case_num:
        train_case.append( sample[i])
        i += 1
    train_case = np.array(train_case)
    return train_case

# return test set with all labels
def case_test(sample, case_num):
    i = 1
    test_case = []
    while i <= case_num:
        test_case.append(sample[-i])
        i += 1
    test_case = np.array(test_case)
    return test_case


def within_class(sample_1, sample_2):
    char_1 = chara(sample_1)
    char_2 = chara(sample_2)
    m_1 = char_1.mean(axis = 0)
    m_2 = char_2.mean(axis = 0)
    # print(len(m_1))
    S_1 = np.zeros((len(m_1), len(m_1)))
    S_2 = np.zeros((len(m_1), len(m_1)))
    for i in range(0, len(char_1)):
        S_1 += np.outer(char_1[i] - m_1, char_1[i] - m_1)
        # print(np.shape(char_1[i] - m_1))
    for i in range(0, len(char_2)):
        S_2 += np.outer(char_2[i] - m_2, char_2[i] - m_2)
    S = S_1 + S_2
    return S

def fisher_train(case, train_num = 600):
    train = case_train(case, train_num)
    X0 = sample_0(train)
    X1 = sample_1(train)
    m_0 = chara(X0).mean(axis = 0)
    m_1 = chara(X1).mean(axis = 0)
    S = within_class(X0, X1)
    w = np.linalg.inv(S) @ (m_1 - m_0).reshape(-1, 1)
    w1 = w / np.linalg.norm(w)
    threshold = (np.mean(chara(X0) @ w1) + np.mean(chara(X1) @ w1)) / 2
    return w1, threshold

w_star, threshold = fisher_train(data)

print('the threshold', threshold)

def fisher_test(case, test_num = 70):
    test = case_test(case, test_num)
    y = label(test)
    X = chara(test)
    projection = X @ w_star
    prediction = (projection > threshold).astype(int)
    accuracy = np.mean(prediction == y)
    return accuracy

print('normalized w*:', w_star)
print('accuracy:', fisher_test(data))



