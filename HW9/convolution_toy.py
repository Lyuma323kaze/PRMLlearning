import numpy as np
import matplotlib.pyplot as plt

A = np.array([
    np.ones(5),
    np.zeros(5),
    np.ones(5),
    np.zeros(5),
    np.ones(5),
])
A = A.T

k1 = np.array([
    -1 * np.ones(3),
    2 * np.ones(3),
    -1 * np.ones(3),
])

k2 = k1.T

def convolution(A_input, kernel, stride):
    shape0 = (A.shape[0] - kernel.shape[0]) // stride + 1
    shape1 = (A.shape[1] - kernel.shape[1]) // stride + 1
    kx = kernel.shape[0]
    ky = kernel.shape[1]
    result = np.empty((shape0, shape1))
    for i in range(shape0):
        for j in range(shape1):
            result[i, j] = (A[i:(i+kx), j:(j+ky)] * kernel).sum()
    return result


stride = 1
result1 = convolution(A, k1, stride)
result2 = convolution(A, k2, stride)

print(result1)
print(result2)

