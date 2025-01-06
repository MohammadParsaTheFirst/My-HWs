import numpy as np
import matplotlib.pyplot as plt

def perceptron(x1, x2, w1=1, w2=1, b=-1.5):
    weighted_sum = w1 * x1 + w2 * x2 + b
    return 1 if weighted_sum >= 0 else 0

# Test the perceptron with all input combinations for an AND gate
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
outputs = [perceptron(x1, x2) for x1, x2 in inputs]

# Print the results
for inp, out in zip(inputs, outputs):
    print(f"Input: {inp} -> Output: {out}")

# Visualize the perceptron decision boundary
plt.figure()

# Create a grid of points to plot the decision boundary
x_values = np.linspace(-0.5, 1.5, 200)
y_values = np.linspace(-0.5, 1.5, 200)
xx, yy = np.meshgrid(x_values, y_values)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = np.array([perceptron(x1, x2) for x1, x2 in grid])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.3)
for x1, x2 in inputs:
    plt.scatter(x1, x2, color='blue' if perceptron(x1, x2) else 'red')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('AND Gate with Perceptron')
plt.grid(True)
plt.show()