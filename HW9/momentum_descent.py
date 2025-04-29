import numpy as np
import matplotlib.pyplot as plt


def grad_L(x):
    if x.shape[0] != 2:
        raise ValueError('x must be a 2-dimensional array')
    grad = 2 * np.array((x[0], 99 * x[1]))
    return grad

def loss(x):
    if x.shape[0] != 2:
        raise ValueError('x must be a 2-dimensional array')
    return x[0] ** 2 + 99 * x[1] ** 2

def gradient_descent(x0, eta, iterations, m = 0.):
    x = np.copy(x0)
    velocity = np.zeros_like(x)
    x_ls = []
    loss_ls = []
    x_ls.append(x)
    loss_ls.append(loss(x))
    value = x.copy()
    for _ in range(iterations):
        grad = grad_L(value)
        velocity = - eta * grad + m * velocity
        value = value + velocity
        x_ls.append(value)
        loss_ls.append(loss(value))
    x_ls = np.array(x_ls)
    return x_ls, loss_ls

eta = 0.02
m = 0.9
iter = 100
lim = 0.01

np.random.seed(941)
x0 = np.random.uniform(low = -lim, high = lim, size = 2)
print(f'x0 = {x0}')

x_simple, loss_simple = gradient_descent(x0, eta, iter)
x_momentum, loss_momentum = gradient_descent(x0, eta, iter, m)

print(f'x_simple: {x_simple[-1]}')
print(f'x_momentum: {x_momentum[-1]}')


color_range = np.arange(len(x_simple))

def plot_trajectory_with_color(ax, trajectory, title):
    n_points = len(trajectory)
    colors = color_range

    sc = ax.scatter(
        trajectory[:, 0],
        trajectory[:, 1],
        c=colors,
        cmap='coolwarm',
        s=20,  # 点的大小
        edgecolor='none'
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Iteration')

    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    for i in range(1, n_points):
        ax.annotate('',
                    xy=trajectory[i],
                    xytext=trajectory[i - 1],
                    arrowprops=dict(arrowstyle="->", color='gray', alpha=0.5))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


plot_trajectory_with_color(ax1, x_simple, f'x_simple Trajectory@eta = {eta}')

plot_trajectory_with_color(ax2, x_momentum, f'x_momentum Trajectory@eta = {eta}')

plt.tight_layout()
plt.savefig(f'trajectory_comparison@eta = {eta}.png')
plt.close()


plt.plot(loss_simple, 'b-o', markersize=4, label='Simple descent')
plt.plot(loss_momentum, 'r-s', markersize=4, label='Momentum descent')

plt.yscale('log')
plt.title(f'Loss Comparison@eta = {eta}')
plt.xlabel('Iteration')
plt.ylabel('Loss (log scale)')
plt.grid(True, which="both", ls='-')
plt.legend()

plt.savefig(f'loss_comparison@eta = {eta}.png', bbox_inches='tight')
plt.close()
