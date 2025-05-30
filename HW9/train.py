import numpy as np


# Define the ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Define the derivative of the ReLU function
def relu_derivative(x):
    return (x > 0).astype(float)

# Define the forward pass of the network
def forward(x, W0, W1, W2, b0, b1, b2):
    z1 = np.dot(x, W0) + b0
    a1 = relu(z1)
    z2 = np.dot(a1, W1) + b1
    a2 = relu(z2)
    z3 = np.dot(a2, W2) + b2
    y_hat = z3
    return y_hat, a1, a2

def compute_loss(x, y, W0, W1, W2, b0, b1, b2):
    y_hat, _, _ = forward(x, W0, W1, W2, b0, b1, b2)
    loss = np.mean((y_hat - y) ** 2) / 2
    return loss
# Define a function to compute gradients by definition
def compute_gradient_definition(x, y, W0, W1, W2, b0, b1, b2, delta=1e-5):
    grads = []
    for param in [W0, W1, W2, b0, b1, b2]:
        param_grad = np.zeros_like(param)
        for idx in np.ndindex(param.shape):
            original_val = param[idx]

            param[idx] = original_val + delta
            y_hat, _, _ = forward(x, W0, W1, W2, b0, b1, b2)
            loss_plus_delta = np.mean((y_hat - y) ** 2) / 2

            param[idx] = original_val - delta
            y_hat, _, _ = forward(x, W0, W1, W2, b0, b1, b2)
            loss_minus_delta = np.mean((y_hat - y) ** 2) / 2

            param_grad[idx] = (loss_plus_delta - loss_minus_delta) / (2 * delta)

            param[idx] = original_val

        grads.append(param_grad)

    return grads


# Define the function for students to implement back-propagation
def compute_gradient(x, y, W0, W1, W2, b0, b1, b2, a1, a2):
    y_hat = np.dot(a2, W2) + b2

    grad_yhat = y_hat - y

    grad_w2 = np.dot(a2.T, grad_yhat) / x.shape[0]
    grad_b2 = np.mean(grad_yhat, axis=0)

    grad_a2 = np.dot(grad_yhat, W2.T) * relu_derivative(a2)

    grad_w1 = np.dot(a1.T, grad_a2) / x.shape[0]
    grad_b1 = np.mean(grad_a2, axis=0)

    grad_a1 = np.dot(grad_a2, W1.T) * relu_derivative(a1)

    grad_w0 = np.dot(x.T, grad_a1) / x.shape[0]
    grad_b0 = np.mean(grad_a1, axis=0)
    # print('the shapes are as below')
    # print(grad_w0.shape, grad_w1.shape, grad_w2.shape, grad_b0.shape, grad_b1.shape, grad_b2.shape)
    return grad_w0, grad_w1, grad_w2, grad_b0, grad_b1, grad_b2

# Dataset generation function
def generate_dataset(num_samples, input_dim):
    X = np.random.randn(num_samples, input_dim)
    y = np.linalg.norm(X, axis=1, ord=2).reshape(-1, 1)
    return X, y


# Gradient descent implementation for a norm prediction regression problem
def gradient_descent(x, y, learning_rate, epochs):
    np.random.seed(42)
    W0 = np.random.randn(x.shape[1], 50).astype(np.float32) / np.sqrt(x.shape[1])
    b0 = np.zeros(50)
    W1 = np.random.randn(50, 50).astype(np.float32) / np.sqrt(50)
    b1 = np.zeros(50)
    W2 = np.random.randn(50, 1).astype(np.float32) / np.sqrt(50)
    b2 = np.zeros(1)

    y_hat, a1, a2 = forward(x, W0, W1, W2, b0, b1, b2)
    grads_bp = compute_gradient(x, y, W0, W1, W2, b0, b1, b2, a1, a2)

    # Compute gradients using definition
    grads_def = compute_gradient_definition(x, y, W0, W1, W2, b0, b1, b2)

    # Print gradients
    for _ in range(len(grads_def)):
        print("Gradients computed using back-propagation: ", grads_bp[_])
        print("Gradients computed using definition: ", grads_def[_])
        print(f"diff {np.abs(grads_def[_] - grads_bp[_]).max()}")

    print("Please make sure all the difference are sufficiently small to go on")

    for epoch in range(epochs):
        # Forward pass
        y_hat, a1, a2 = forward(x, W0, W1, W2, b0, b1, b2)

        # Compute gradients using back-propagation
        grads_bp = compute_gradient(x, y, W0, W1, W2, b0, b1, b2, a1, a2)

        print(f"{epoch}: loss is {compute_loss(x, y, W0, W1, W2, b0, b1, b2)}")

        # Update parameters using gradients from back-propagation
        W0 -= learning_rate * grads_bp[0]
        W1 -= learning_rate * grads_bp[1]
        W2 -= learning_rate * grads_bp[2]
        b0 -= learning_rate * grads_bp[3]
        b1 -= learning_rate * grads_bp[4]
        b2 -= learning_rate * grads_bp[5]

    return W0, W1, W2, b0, b1, b2


# Generate dataset
X_train, y_train = generate_dataset(500, 10)

# Set learning rate and number of epochs
learning_rate = 1e-2
epochs = 100

# Train the network using gradient descent
W0, W1, W2, b0, b1, b2 = gradient_descent(X_train, y_train, learning_rate, epochs)

X_test, y_test = generate_dataset(100, 10)
test_loss = compute_loss(X_test, y_test, W0, W1, W2, b0, b1, b2)
print(f"Test loss is {test_loss}")