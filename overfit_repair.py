import random
import math
import matplotlib.pyplot as plt

# 1. Create messy data (XOR with noise)
X = [[0,0], [0,1], [1,0], [1,1]] * 10
Y = [0, 1, 1, 0] * 10
X = [[x[0] + random.uniform(-0.1, 0.1), x[1] + random.uniform(-0.1, 0.1)] for x in X]

def train(use_reg=False):
    # Initialize weights
    w = [random.uniform(-1, 1) for _ in range(2)]
    loss_history = []
    
    for epoch in range(200):
        total_err = 0
        for i in range(len(X)):
            # Predict
            dot = X[i][0] * w[0] + X[i][1] * w[1]
            pred = 1 / (1 + math.exp(-max(min(dot, 50), -50)))
            
            # Error
            err = Y[i] - pred
            total_err += err**2
            
            # Update weights
            reg_term = 0.1 * w[0] if use_reg else 0 # This is L2!
            w[0] += (err * X[i][0] - reg_term) * 0.1
            w[1] += (err * X[i][1] - reg_term) * 0.1
            
        loss_history.append(total_err/len(X))
    return loss_history

# Run and Plot
plt.plot(train(use_reg=False), label='Overfitting (No Fix)')
plt.plot(train(use_reg=True), label='Regularization (Fixed)')
plt.legend()
plt.title("Mission 1: Fixing Overfitting")
plt.savefig('overfitting_result.png')
plt.show()