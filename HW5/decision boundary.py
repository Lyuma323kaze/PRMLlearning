import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# 示例函数（根据你的实际函数修改）
def decision_function(x, y):
    return np.exp(6 * x + 2 * y - 10) - 4  # 例如，决策边界是 x + 2y - 1 = 0

# 生成数据网格
x_min, x_max = -30, 30
y_min, y_max = -30, 30
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 2000),
                     np.linspace(y_min, y_max, 2000))

# 计算每个网格点的决策值
Z = decision_function(xx, yy)

# 创建画布
plt.figure(figsize=(8, 6))

# 绘制决策边界线（Z=0的等高线）
contour = plt.contour(xx, yy, Z, levels=[0], linewidths=2, linestyles='--')
plt.clabel(contour, inline=True, fontsize=12)  # 添加等高线数值标签

# 根据Z的正负填充不同颜色，alpha为透明度
plt.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf],
             colors=['#FFAAAA', '#AAFFAA'], alpha=0.3)

# 手动创建图例元素（兼容最新版本）
legend_elements = [
    Patch(facecolor='#FFAAAA', edgecolor='none', alpha=0.3, label='omega_1'),
    Patch(facecolor='#AAFFAA', edgecolor='none', alpha=0.3, label='omega_2'),
    Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Decision Boundary')
]

# 添加图例和标签
plt.legend(handles=legend_elements, loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Boundary Visualization')
plt.grid(True)
plt.show()