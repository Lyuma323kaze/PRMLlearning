import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Target distribution parameters
mu_target = np.array([1.0, 2.0])
sigma_target = np.diag([2.0, 1.0])  # Diagonal covariance

# Initial learned parameters
mu = np.array([0.0, 0.0])
sigma_diag = np.array([1.0, 1.0])  # Diagonal variances

# Learning rate and number of steps
eta = 0.1
steps = 50

# Store history for visualization
history_mu = [mu.copy()]
history_sigma = [sigma_diag.copy()]

# Gradient descent updates
for _ in range(steps):
    # Compute gradients for mu
    grad_mu = (mu - mu_target) / np.diag(sigma_target)
    
    # Compute gradients for sigma (diagonal)
    grad_sigma_diag = 0.5 * (1/np.diag(sigma_target) - 1/sigma_diag)
    
    # Update parameters
    new_mu = mu - eta * grad_mu
    new_sigma_diag = sigma_diag - eta * grad_sigma_diag
    
    # Ensure variances remain positive
    new_sigma_diag = np.maximum(new_sigma_diag, 1e-3)
    
    # Update parameters
    mu = new_mu
    sigma_diag = new_sigma_diag
    
    # Save history
    history_mu.append(mu.copy())
    history_sigma.append(sigma_diag.copy())

# Visualization
x = np.linspace(-1, 3, 100)
y = np.linspace(-1, 4, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

fig, ax = plt.subplots(figsize=(10, 8))

# Plot target distribution
target_dist = multivariate_normal(mean=mu_target, cov=sigma_target)
Z_target = target_dist.pdf(pos)
ax.contour(X, Y, Z_target, levels=5, colors='red', label='Target')

# Plot initial distribution
initial_dist = multivariate_normal(mean=history_mu[0], cov=np.diag(history_sigma[0]))
Z_initial = initial_dist.pdf(pos)
ax.contour(X, Y, Z_initial, levels=5, colors='blue', linestyles='dashed', label='Initial')

# Plot intermediate steps
for step in [10, 20, 30, 40, 50]:
    mu_step = history_mu[step]
    sigma_step = np.diag(history_sigma[step])
    dist = multivariate_normal(mean=mu_step, cov=sigma_step)
    Z = dist.pdf(pos)
    ax.contour(X, Y, Z, levels=5, colors='green', alpha=0.3)

# Plot final distribution
final_dist = multivariate_normal(mean=history_mu[-1], cov=np.diag(history_sigma[-1]))
Z_final = final_dist.pdf(pos)
ax.contour(X, Y, Z_final, levels=5, colors='green', label='Final')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Evolution of Learned Gaussian Distribution')
plt.savefig('prob1.png')