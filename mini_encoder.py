import numpy as np
import matplotlib.pyplot as plt

# 1. Create Clustered Data
data = np.vstack([
    np.random.normal(0.2, 0.05, (50, 2)),
    np.random.normal(0.8, 0.05, (50, 2))
])

def run_autoencoder(bottleneck_size):
    # Architecture: 2 -> bottleneck -> 2
    W_enc = np.random.randn(2, bottleneck_size)
    W_dec = np.random.randn(bottleneck_size, 2)
    losses = []

    for epoch in range(300):
        # Forward (Encoder -> Decoder)
        latent = np.dot(data, W_enc) 
        output = np.dot(latent, W_dec)
        
        # MSE Loss
        loss = np.mean((output - data)**2)
        losses.append(loss)
        
        # Backprop (Simple Linear Gradient)
        error = output - data
        dW_dec = np.dot(latent.T, error)
        dW_enc = np.dot(data.T, np.dot(error, W_dec.T))
        
        W_enc -= 0.01 * dW_enc
        W_dec -= 0.01 * dW_dec
    return losses, latent

# Compare Bottleneck sizes
loss_1, latent_1 = run_autoencoder(1) # High compression
loss_2, latent_2 = run_autoencoder(2) # Low compression

plt.plot(loss_1, label='Bottleneck Size 1')
plt.plot(loss_2, label='Bottleneck Size 2')
plt.title("Reconstruction Loss")
plt.legend()
plt.show()
import math
import random

# Our simple data
points = [[0.1, 0.9], [0.2, 0.8], [0.8, 0.1], [0.9, 0.2]]

# Shrink (Encode) and Grow (Decode)
for epoch in range(300):
    for p in points:
        # Step 1: Shrink to 1 number (The Bottleneck)
        latent = (p[0] * 0.5) + (p[1] * 0.5) 
        
        # Step 2: Try to grow back to 2 numbers
        guess = [latent * 1.0, latent * 1.0]
        
        # Step 3: See how wrong we were (Loss)
        error = ((p[0] - guess[0])**2 + (p[1] - guess[1])**2) / 2

print("Autoencoder finished its game of Telephone!")