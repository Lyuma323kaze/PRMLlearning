import numpy as np
import matplotlib.pyplot as plt
import os

data = np.zeros([1000, 2])
data[:, 0] = np.random.choice([0, 1], size = 1000, p = [1 - 1/3, 1/3])
data = np.array(data)

mean1 = [0, 0]
mean2 = [3, 1]

cov = np.array([[1, 0], [0, 1]])
kernel = ""

for i in range(len(data)):
    if data[i][0] == 0:
        data[i] = np.random.multivariate_normal(mean1, cov, 1).flatten()
    else:
        data[i] = np.random.multivariate_normal(mean2, cov, 1).flatten()

def gene_range(left, right, h):
    return np.arange(left, right + h, h), np.arange(left, right + h, h)


h_list = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

h = None
left = -3
right = 5

def ker_parzen(x, y, h):
    diff = x - y
    norms = np.linalg.norm(diff, ord = np.inf, axis = -1)
    return np.where(norms <= h, 1, 0), "parzen"

def ker_gaussian(x, y, h):
    diff = x - y
    return (1 / np.sqrt(2 * np.pi * h ** 2)) * np.exp(- np.linalg.norm(diff, axis = 2)/ (2 * h ** 2)), "gaussian"

def ker_exp(x ,y, h):
    diff = x - y
    return (1 / h) * np.exp(- np.linalg.norm(diff, axis = 2) / h), "exp"

def compute_density(data, ker, h):
    global kernel
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    data_exp = data[:, np.newaxis, :]
    grid_exp = grid_points[np.newaxis, :, :]

    kernel_values, kernel = ker(data_exp, grid_exp, h)

    z = kernel_values.sum(axis = 0)
    z = z.reshape(xx.shape)
    z = z / (data.shape[0] * h ** 2)
    return z

def prob(x, y):
    return (1 / (2 * np.pi)) * ((2 / 3) * np.exp(- (x ** 2 + y ** 2) / 2)
                              + (1 - 2 / 3) * np.exp(- ((x - 3) ** 2 + (y - 1) ** 2) / 2))

def compute_error(prob_func, fit_prob, h):
    xx, yy = np.meshgrid(x_range, y_range)
    f = prob_func(xx, yy)
    err = (np.abs(f - fit_prob)).sum() * h ** 2
    return err

err_ls = []
for i in range(len(h_list)):
    h = h_list[i]
    x_range, y_range = gene_range(left, right, h)
    z = compute_density(data, ker_exp, h)
    mxm = z.max()
    err = compute_error(prob, z, h)
    err_ls.append(err)
    print(f'error = {err}, h = {h}, ker = {kernel}')

    file_folder = 'results'
    if not os.path.exists(file_folder):
        os.makedirs(file_folder)
    if not os.path.exists(os.path.join(file_folder, f'@{kernel}')):
        os.makedirs(os.path.join(file_folder, f'@{kernel}'))
    file_subfolder = os.path.join(file_folder, f'@{kernel}')

    plt.imshow(z, extent=(-3, 5, -3, 5),
               vmin=0, vmax=mxm, cmap='coolwarm', origin='lower')
    plt.colorbar(label='Density Value')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Heatmap of Density, h = {h}, ker = {kernel}')

    file_path = os.path.join(file_subfolder, f'Result at {h:.2f}@error = {err}.png')
    plt.savefig(file_path)
    plt.close()
err_ls = np.array(err_ls)
print(f'The best composition of kernel {kernel} is: \n h = {h_list[np.argmin(err_ls)]}, error = {err_ls.min()} ')