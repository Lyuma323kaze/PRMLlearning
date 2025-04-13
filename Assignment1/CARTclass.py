from math import log10
from collections import Counter
import numpy as np
from skimage.filters.rank import threshold


class CARTDecisionTree:
    def __init__(self, max_depth = None):
        self.tree = None
        self.max_depth = max_depth
    class Node:
        def __init__(self, feat_idx = None, children = None, leaf_label = None, threshold = None, pm = None):
            self.featidx = feat_idx
            self.children = children
            self.lflabel = leaf_label   # None: not leaf; False
            self.threshold = threshold
            self.pm = pm    # 1 means greater than threshold, 0 vise versa

    def _loss_single_feature(self, x: np.array, y):
        l_ls = []
        v_ls = []
        thresholds = np.zeros(len(x) + 1)
        thresholds[:-1] = x
        thresholds[-1] = max(x) + 1
        for threshold in thresholds:
            mask = np.array(x >= threshold)
            mean_mask = mask.astype(int).mean()
            if mean_mask != 0 and mean_mask != 1:
                v1 = y[mask].mean()     # v1 is larger than threshold, vise versa
                v2 = y[~mask].mean()
                loss = ((y[mask] - v1) ** 2).sum() + ((y[~mask] - v2) ** 2).sum()
                l_ls.append(loss)
                v_ls.append((v1 >= 0.5, v2 >= 0.5))
            elif mean_mask == 0:
                v1 = y.mean()
                v2 = 2      # invalid symbol
                loss = ((y[~mask] - v1) ** 2).sum()
                l_ls.append(loss)
                v_ls.append((v1 >= 0.5, v2))
            elif mean_mask == 1:
                v1 = 2      # invalid symbol
                v2 = y.mean()
                loss = ((y[~mask] - v1) ** 2).sum()
                l_ls.append(loss)
                v_ls.append((v1, v2 >= 0.5))    # if all are smaller than the threshold, v1 would be None, v2 is the overall mean

        idx = np.argmin(l_ls)
        return thresholds[idx], v_ls[idx]

    def _gini(self, y):
        values = np.unique(y)
        length = len(y)
        gini = 1
        for value in values:
            mask = np.array(y == value)
            count = mask.sum()
            gini -= (count / length) ** 2
        return gini

    def _gini_gain_on_feature(self, x, threshold, y):
        mask = np.array(x >= threshold)
        D1 = y[mask]
        D2 = y[~mask]
        return self._gini(y) - (len(D1) / len(y)) * self._gini(D1) - (len(D2) / len(y)) * self._gini(D2)

    def _best_feature_CART(self, X, y, feat_idx):
        gains = [self._gini_gain_on_feature(X[:,i],
                                            (self._loss_single_feature(X[:, i], y))[0],
                                            y)
                for i in feat_idx]
        best_idx = np.argmax(gains)
        # v_tuple[0] related to >= threshold, vise versa
        threshold, v_tuple = self._loss_single_feature(X[:, best_idx], y)   # when all are smaller than threshold, v_tuple[0] is False
        return feat_idx[best_idx], threshold, v_tuple

    def _build_tree(self, data_X, y, feat_idx, depth = 0, pm = None):
        # print(f'length = {len(data_X)}, depth = {depth}, pm = {pm}')
        if len(np.unique(y)) == 1:
            return self.Node(leaf_label = y[0])     # pure node
        if (self.max_depth and depth >= self.max_depth):
            main_class = Counter(y).most_common(1)[0][0]
            return self.Node(leaf_label = main_class)       # deepest node
        best_feat_idx, threshold, v_tuple = self._best_feature_CART(data_X, y, feat_idx)
        best_feat_value = data_X[:, best_feat_idx]

        children = [None, None]

        mask = np.array(best_feat_value >= threshold)
        X_left = data_X[mask]
        y_left = y[mask]
        X_right = data_X[~mask]
        y_right = y[~mask]

        # >= threshold
        if len(y_left) == 0:
            children[0] = self.Node(leaf_label = v_tuple[0], threshold = threshold, pm = 1)     # node with label False, no element in
        else:
            children[0] = self._build_tree(X_left, y_left, feat_idx, depth + 1, pm = 1)

        # < threshold
        if len(y_right) == 0:
            children[1] = self.Node(leaf_label = v_tuple[1], threshold = threshold, pm = 0)
        else:
            children[1] = self._build_tree(X_right, y_right, feat_idx, depth + 1, pm = 0)

        return self.Node(feat_idx = best_feat_idx, children = children, threshold = threshold, pm = pm)

    def train(self, data_X, y):
        feat_idx = list(range(data_X.shape[1]))
        self.tree = self._build_tree(data_X, y, feat_idx)

    # vectorize can be added
    def _predict_sample(self, data_x, node):
        if (node.lflabel is not None):
            return node.lflabel

        feature = data_x[node.featidx]
        if node.threshold is not None:
            # where to vectorize
            if feature >= node.threshold:
                return self._predict_sample(data_x, node.children[0])
            elif feature < node.threshold:
                return self._predict_sample(data_x, node.children[1])

    def compute_accuracy(self, X, y):
        prediction = self.predict(X)
        if y is not None:
            mask = np.array(prediction == y)
            accuracy = mask.astype(int).mean()
            return accuracy, mask, prediction
        else:
            return None, None, prediction

    def predict(self, data_X):
        return np.array([self._predict_sample(data_x, self.tree) for data_x in data_X])

    def print_tree(self, node = None, indent = 0, feature_names = None):
        if node is None:
            if self.tree is None:
                print('not trained yet')
                return
            node = self.tree

        if node.lflabel is not None:
            print(" " * indent + f'Label:{node.lflabel}, pm:{node.pm}, threshold:{node.threshold}')
            return

        feat_name = f'Chara[{node.featidx}]' if feature_names is None else f'Chara:{feature_names[node.featidx]}'
        print(" " * indent + f'{feat_name}')
        for value, child in node.children.items():
            print(" " * (indent + 2) + f"Value = {value} and")
            self.print_tree(child, indent + 4, feature_names)

    '''
        def _entropy(self, y):
            counts = Counter(y)
            fracs = [count / len(y) for count in counts.values()]
            return - sum(p * log10(p) for p in fracs)
    '''
    '''
        def _best_feature(self, data_X, y, feat_idx):
            gains = [self._infor_gain(data_X[:, i], y) for i in feat_idx]
            best_idx = np.argmax(gains)
            return feat_idx[best_idx]
    '''
    '''
        def _infor_gain(self, chara, y):
            entropy_before = self._entropy(y)
            unique_values = set(chara)
            entropy_loss = 0
            for value in unique_values:
                mask = (chara == value)
                subset_y = y[mask]
                weight = len(subset_y) / len(y)
                entropy_loss += weight * self._entropy(subset_y)
            return entropy_before - entropy_loss
    '''