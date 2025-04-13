from adaboostClass import Adaboost
import numpy as np

np.set_printoptions(precision = 4)

a_range = (1, 10)
b_range = (2, 14)
def gene_clsfers(ran, idx):
    clsfers = []
    for i in range(ran[0], ran[1] + 1):
        def make_function(i):
            func = lambda x: (1, f'index {idx} = {i}') if x[idx] > i \
                else (-1, f'index {idx} = {i}')
            return func
        clsfers.append(make_function(i))
    return clsfers

data = np.array([
    [1, 10, -1],
    [3, 11, -1],
    [4, 4, -1],
    [4, 2, -1],
    [5, 10, 1],
    [5, 6, 1],
    [7, 7, 1],
    [8, 8, -1],
    [8, 7, 1],
    [10, 14, 1]
])

all_clsfers = gene_clsfers(a_range, 0) + gene_clsfers(b_range, 1)

case = Adaboost(all_clsfers, 25)
case.train_weak(data[:, :-1], data[:, -1], all_clsfers)

