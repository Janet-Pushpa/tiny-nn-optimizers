import math
import random
import matplotlib.pyplot as plt

# 1. Special starting numbers (He Initialization)
def he_init(n_in):
    return random.gauss(0, math.sqrt(2/n_in))

# 2. A Deep Brain (3 Hidden Layers)
# Layers: 2 inputs -> 3 neurons -> 3 neurons -> 3 neurons -> 1 output
W1 = [[he_init(2) for _ in range(3)] for _ in range(2)]
W2 = [[he_init(3) for _ in range(3)] for _ in range(3)]
W3 = [[he_init(3) for _ in range(3)] for _ in range(3)]
W4 = [[he_init(3) for _ in range(1)] for _ in range(3)]

# Now you just run your training loop like before!
print("Deep Brain is ready and stable!")