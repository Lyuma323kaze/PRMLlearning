import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
set = [[-3, 1], [-2, 1], [-1, -1], [0, -1], [1, -1], [2, 1], [3, 1]]
set = np.array(set)
w = cp.Variable(2)
b = cp.Variable()
objective = cp.Minimize(cp.norm(w,2)**2)
def phi(point):
    point = np.array(point)
    if point.ndim == 1:
        return np.array([point[0],point[0] ** 2])
    else:
        return np.column_stack([point[:, 0], point[:, 0] ** 2])
constraints = [set[i][1] * (w.T @ phi(set[i]) + b) >= 1 for i in range(set.shape[0])]
prob = cp.Problem(objective, constraints)

prob.solve()

w_star = w.value
b_star = b.value
margin = 2 / np.linalg.norm(w_star)
print('margin =',margin)

support_vector_indices = np.where(np.abs(set[:, 1] * (phi(set).dot(w_star) + b_star) - 1) <= 1e-5)[0]
support_vectors = set[support_vector_indices]
print("support vectors:", support_vectors)

plt.figure(1)
values = set[:, 1]
x_range = np.linspace(set[:, 0].min(), set[:, 0].max(), 100)
if w_star[1] != 0:
    line = (- b_star - w_star[0] * x_range) / w_star[1]
    plt.plot(x_range, line, label = 'Optimal Decision Boundary')
else:
    x_line = ( - b_star) / w_star[0]
    plt.plot(x = x_line, label = 'Optimal Decision Boundary')
plt.scatter(phi(set)[:, 0], phi(set)[:, 1], c = values, cmap = "coolwarm")
plt.colorbar(label = 'Value')
plt.title("Training Data")
plt.xlabel("x_i")
plt.ylabel("x_i**2")
plt.legend(loc = 3)
plt.grid()
# plt.savefig('Problem1(1).png')

plt.figure(2)
line_2 = - w_star[0] * x_range - w_star[1] * x_range ** 2 - b_star
plt.plot(x_range, line_2, label = "Decision Boundary")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Boundary on xy plane")
plt.grid()
plt.scatter(set[:, 0], set[:, 1], c = values, cmap = "coolwarm")
plt.colorbar(label = 'Value')
plt.legend(loc = 3)
plt.savefig('Problem1(2).png')

plt.show()
