from main import *
from weak_classifiers import linear_clsfer_fisher

X_test = data_test[:, :-1]
y_test = data_test[:, -1]

acc_linear = linear(X_test = X_test, y_test = y_test, _data_train = data_train)[0]
acc_linear_fisher = linear_clsfer_fisher(X_test = X_test, y_test = y_test, _data_train = data_train)[0]
acc_tree = tree(X_test = X_test, y_test = y_test, _data_train = data_train)[0]

print(f'accuracy of linear classifier: {acc_linear}')
print(f'accuracy of tree classifier: {acc_tree}')
