import numpy as np
from CARTclass import CARTDecisionTree

def linear_clsfer_fisher(X_test = None,
                         data_train = None,
                         y_test = None,
                         train_size=0.7,
                         test_size=0.3,
                         compute = True,
                         para = None,
                         predict = False):
    def fisher_train(data_train):
        mask_0 = (data_train[:, -1] == 0)
        X0 = data_train[mask_0, :-1]
        X1 = data_train[~mask_0, :-1]
        m_0 = X0.mean(axis = 0)
        m_1 = X1.mean(axis = 0)
        diff0 = X0 - m_0
        diff1 = X1 - m_1
        S = np.einsum('ki,kj->ij', diff0, diff0) + np.einsum('ki,kj->ij', diff1, diff1)
        w = np.linalg.inv(S) @ (m_1 - m_0).reshape(-1, 1)
        w1 = w / np.linalg.norm(w)
        threshold = (np.mean(X0 @ w1) + np.mean(X1 @ w1)) / 2
        return w1, threshold

    def fisher_test(X_test, threshold, w_star, label = None):
        X = X_test
        y = label
        projection = X @ w_star
        prediction = np.array(projection > threshold).astype(int)
        if y is not None:
            mask = (prediction == y)
            accuracy = np.mean(prediction == y)
            return accuracy, mask, prediction
        else:
            return None, None, prediction

    if para is None:
        if data_train is None:
            raise ValueError("data_train is None")
        w_star, threshold = fisher_train(data_train)
    else:
        w_star, threshold = para

    if predict:
        return fisher_test(X_test, threshold, w_star)[2]

    if compute:
        accuracy, mask, prediction = fisher_test(X_test, threshold, w_star, label = y_test)
        return accuracy, mask
    else:
        return w_star, threshold

def tree_clsfer(X_test = None,
                data_train = None,
                y_test = None,
                para = None,
                train_size=0.7,
                test_size=0.3,
                compute = True,
                predict = False):
        toy = CARTDecisionTree(2)

        if para is None:
            if data_train is None:
                raise ValueError("data_train is None")
            toy.train(data_train[:, :-1], data_train[:, -1])
        else:
            toy = para

        if predict:
            return toy.compute_accuracy(X_test, y_test)[2]

        if compute:
            acc, mask, prediction = toy.compute_accuracy(X_test, y_test)
            return acc, mask
        else:
            return toy



