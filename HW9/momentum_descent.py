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
iter = 20
lim = 0.01

np.random.seed(941)
x0 = np.random.uniform(low = -lim, high = lim, size = 2)
print(x0)

x_simple, loss_simple = gradient_descent(x0, eta, iter)
x_momentum, loss_momentum = gradient_descent(x0, eta, iter, m)

color_range = np.arange(len(x_simple))


def plot_trajectory_with_color(ax, trajectory, title):
    n_points = len(trajectory)
    colors = color_range

    # 绘制散点图（颜色表示迭代顺序）
    sc = ax.scatter(
        trajectory[:, 0],
        trajectory[:, 1],
        c=colors,
        cmap='viridis',  # 选择颜色映射
        s=20,  # 点的大小
        edgecolor='none'
    )

    # 添加颜色条
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Iteration')

    # 添加标题和标签
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    # 可选：添加箭头表示方向
    for i in range(1, n_points):
        ax.annotate('',
                    xy=trajectory[i],
                    xytext=trajectory[i - 1],
                    arrowprops=dict(arrowstyle="->", color='gray', alpha=0.5))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 绘制x_simple的轨迹
plot_trajectory_with_color(ax1, x_simple, 'x_simple Trajectory')

# 绘制x_momentum的轨迹
plot_trajectory_with_color(ax2, x_momentum, 'x_momentum Trajectory')

plt.tight_layout()
plt.savefig('trajectory_comparison.png')
plt.close()

plt.figure()
plt.plot(x_simple[:, 0], x_simple[:, 1])
plt.title('x_simple')
plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig('x_simple.png')

plt.figure()
plt.plot(x_momentum[:, 0], x_momentum[:, 1])
plt.title('x_momentum')
plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig('x_momentum.png')

plt.figure()
plt.plot(loss_simple)
plt.title('loss_simple')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.savefig('loss_simple.png')

plt.figure()
plt.plot(loss_momentum)
plt.title('loss_momentum')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.savefig('loss_momentum.png')
