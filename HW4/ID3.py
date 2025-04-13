import numpy as np
from ID3class import ID3DecisionTree


data = np.array([
    ["Sunny", "Hot", "High", "Weak", "No"],
    ["Sunny", "Hot", "High", "Strong", "No"],
    ["Overcast", "Hot", "High", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Strong", "No"],
    ["Overcast", "Cool", "Normal", "Strong", "Yes"],
    ["Sunny", "Mild", "High", "Weak", "No"],
    ["Sunny", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "Normal", "Weak", "Yes"],
    ["Sunny", "Mild", "Normal", "Strong", "Yes"],
    ["Overcast", "Mild", "High", "Strong", "Yes"],
    ["Overcast", "Hot", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Strong", "No"]
])

data_D = np.array([
    ["D1", "Sunny", "Hot", "High", "Weak", "No"],
    ["D2", "Sunny", "Hot", "High", "Strong", "No"],
    ["D3", "Overcast", "Hot", "High", "Weak", "Yes"],
    ["D4", "Rain", "Mild", "High", "Weak", "Yes"],
    ["D5", "Rain", "Cool", "Normal", "Weak", "Yes"],
    ["D6", "Rain", "Cool", "Normal", "Strong", "No"],
    ["D7", "Overcast", "Cool", "Normal", "Strong", "Yes"],
    ["D8", "Sunny", "Mild", "High", "Weak", "No"],
    ["D9", "Sunny", "Cool", "Normal", "Weak", "Yes"],
    ["D10", "Rain", "Mild", "Normal", "Weak", "Yes"],
    ["D11", "Sunny", "Mild", "Normal", "Strong", "Yes"],
    ["D12", "Overcast", "Mild", "High", "Strong", "Yes"],
    ["D13", "Overcast", "Hot", "Normal", "Weak", "Yes"],
    ["D14", "Rain", "Mild", "High", "Strong", "No"]
])

data_M = np.array([
    ["Sunny", "Hot", "High", "Weak", "No"],
    ["Sunny", "Hot", "High", "Strong", "No"],
    ["Rain", "Mild", "High", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Strong", "No"],
    ["Overcast", "Cool", "Normal", "Strong", "Yes"],
    ["Sunny", "Mild", "High", "Weak", "No"],
    ["Sunny", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "Normal", "Weak", "Yes"],
    ["Sunny", "Mild", "Normal", "Strong", "Yes"],
    ["Overcast", "Mild", "High", "Strong", "Yes"],
    ["Overcast", "Hot", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Strong", "No"]
])

tree_D = ID3DecisionTree(1)
tree_D.train(data_D[:, :-1], data_D[:, -1])

# test on data_D
# best_idx_D = tree_D._best_feature(data_D[:, :-1], data_D[:, -1], [i for i in range(data_D.shape[1] - 1)])
# print(best_idx_D)

# on data
data_obj = ID3DecisionTree(10)
data_obj.train(data[:, :-1], data[:, -1])
tree = data_obj.tree
feature_names = np.array(['Outlook', 'Temperature', 'Humidity', 'Wind'])
# data_obj.print_tree(node = tree, feature_names = feature_names)



#on data with missing D3
data_mobj = ID3DecisionTree(10)
data_mobj.train(data_M[:, :-1], data_M[:, -1])
tree_m = data_mobj.tree
data_mobj.print_tree(node = tree_m, feature_names = feature_names)

