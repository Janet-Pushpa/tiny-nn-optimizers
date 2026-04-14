import numpy as np
import matplotlib.pyplot as plt
import os
def sinu_relu(x):
    """
    Sinu-ReLU: f(x) = max(0, x) + sin(x) * exp(-abs(x))
    """
    return np.maximum(0, x) + np.sin(x) * np.exp(-np.abs(x))

def relu(x):
    return np.maximum(0, x)

x = np.linspace(-5, 5, 400)
y_sinu = sinu_relu(x)
y_relu = relu(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y_sinu, label='Sinu-ReLU', linewidth=2)
plt.plot(x, y_relu, label='ReLU', linestyle='--', alpha=0.7)
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.title('Custom Activation Function: Sinu-ReLU')
plt.xlabel('Input (x)')
plt.ylabel('Output (f(x))')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)

# Replace your current 'try' block with this:
try:
    # This ignores the complex path and just saves it in the current working folder
    plt.savefig('sinu_relu_graph.png') 
    plt.show()
    print(f"Success! Graph saved at: {os.getcwd()}\\sinu_relu_graph.png")
except Exception as e:
    print(f"Error: {e}")