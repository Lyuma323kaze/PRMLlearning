import numpy as np
from math import log10
from collections import Counter

class ID3DecisionTree:
    def __init__(self, max_depth = None):
        self.tree = None
        self.max_depth = max_depth
    class Node:
        def __init__(self, feat_idx = None, children = None, leaf_label = None):
            self.featidx = feat_idx
            self.children = children
            self.lflabel = leaf_label

    def _entropy(self, y):
        counts = Counter(y)
        fracs = [count / len(y) for count in counts.values()]
        return - sum(p * log10(p) for p in fracs)

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

    def _best_feature(self, data_X, y, feat_idx):
        gains = [self._infor_gain(data_X[:, i], y) for i in feat_idx]
        best_idx = np.argmax(gains)
        return feat_idx[best_idx]

    def _build_tree(self, data_X, y, feat_idx, depth = 0):
        if len(np.unique(y)) == 1:
            return self.Node(leaf_label = y[0])
        if len(feat_idx) == 0 or (self.max_depth and depth >= self.max_depth):
            main_class = Counter(y).most_common(1)[0][0]
            return self.Node(leaf_label = main_class)
        best_feat_idx = self._best_feature(data_X, y, feat_idx)
        best_feat_charas = np.unique(data_X[:, best_feat_idx])
        remain_features = [f for f in feat_idx if f != best_feat_idx]

        children = {}
        for chara in best_feat_charas:
            mask = (data_X[:, best_feat_idx] == chara)
            if mask.sum() == 0:
                main_class = Counter(y).most_common(1)[0][0]
                children[chara] = self.Node(leaf_label = main_class)
            else:
                subtree = self._build_tree(data_X[mask], y[mask], remain_features, depth + 1)
                children[chara] = subtree

        return self.Node(feat_idx = best_feat_idx, children = children)

    def train(self, data_X, y):
        feat_idx = list(range(data_X.shape[1]))
        self.tree = self._build_tree(data_X, y, feat_idx)

    def _predict_sample(self, data_x, node):
        if node.lflabel is not None:
            return node.lflabel

        feat_chara = data_x[node.lflabel]
        if feat_chara not in node.children:
            return Counter([leaf.lflabel for leaf in node.children.values().most_common(1)[0][0]])

        return self._predict_sample(data_x, node.children[feat_chara])

    def predict(self, data_X):
        return np.array([self._predict_sample(data_x, self.tree) for data_x in data_X])

    def print_tree(self, node = None, indent = 0, feature_names = None):
        if node is None:
            if self.tree is None:
                print('not trained yet')
                return
            node = self.tree

        if node.lflabel is not None:
            print(" " * indent + f'Label:{node.lflabel}')
            return

        feat_name = f'Chara[{node.featidx}]' if feature_names is None else f'Chara:{feature_names[node.featidx]}'
        print(" " * indent + f'{feat_name}')
        for value, child in node.children.items():
            print(" " * (indent + 2) + f"Value = {value} and")
            self.print_tree(child, indent + 4, feature_names)




