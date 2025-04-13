import numpy as np
from pprint import pprint

class Adaboost:
    def __init__(self, clsfers, max_step = 0):
        self.tree = None
        self.max_step = max_step
        self.clsfers = clsfers

    def _compute_weighted_error(self, data_X, y, wk_clsfer, w):
        mask = np.array(([wk_clsfer(row)[0] for row in data_X] == y))
        mask = mask.astype(int)
        epsilon = np.dot(w, mask)
        return epsilon

    def _compute_classifier_weight(self, epsilon):
        return 0.5 * np.log((1 - epsilon) / epsilon)

    def _final_clsfer(self, alpha, wk_clsfers, x):
        H = 0
        for m in alpha.shape[0]:
            H += alpha[m] * wk_clsfers[m](x)[0]
        return np.sign(H)

    def _judge_clsfer(self, data_X, y):
        for clsfer in self.clsfers:
            mask = [([clsfer(row)[0] for row in data_X] == y)]
            if np.sum(mask) / len(mask) <= 0.5:
                self.clsfers.remove(clsfer)
        return len(self.clsfers)


    def train_weak(self, data_X, y, wk_clsfers):
        clsfer_len = self._judge_clsfer(data_X, y)
        print(f'Number of classifiers: {clsfer_len}')
        clsfer_copy = wk_clsfers.copy()
        w = np.ones(data_X.shape[0]) / data_X.shape[0]
        epsilon = []
        alpha = []
        for m in range(min(self.max_step, len(wk_clsfers))):
            wk_clsfer = wk_clsfers[m]
            epsilon_m = self._compute_weighted_error(data_X, y, wk_clsfer, w)
            if epsilon_m >= 0.5:
                clsfer_copy.remove(wk_clsfer)
                continue
            alpha_m = self._compute_classifier_weight(epsilon_m)
            epsilon.append(epsilon_m)
            alpha.append(alpha_m)
            print(f'Step = {m + 1}')

            for i in range(data_X.shape[0]):
                w[i] = w[i] * np.exp(- alpha_m * y[i] * wk_clsfer(data_X[i, :])[0])
            # normalization
            w = w / np.sum(w)
            print(f'Weight w = {w}')
        epsilon = np.array(epsilon)
        alpha = np.array(alpha)
        print(f'epsilon set: {epsilon}')
        print(f'alpha set: {alpha}')
        print(f'Consistency: {(len(alpha) == len(clsfer_copy))}')
        pprint([clsfer_copy[j](data_X[1, :])[1] for j in range(len(clsfer_copy))])
        def F_clsfer(x):
            return self._final_clsfer(alpha, clsfer_copy, x)
        return F_clsfer
