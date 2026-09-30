# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
def ReLU(A):
    return np.maximum(0,A)

def Sigmoid(A):
    return 1/(1+np.exp(-A))

def ForwardPass(x,w1,b1,w2,b2):
    H1 = np.dot(w1, x) + b1
    H1 = ReLU(H1)
    H2 = np.dot(w2, H1) + b2
    H2 = Sigmoid(H2)

    return H2

# %%
# x = np.array([
#     [1,1],
#     [0,0],
#     [1,0],
#     [0,1]
# ])

# y = np.array([
#     [1],
#     [1],
#     [0],
#     [0]
# ])

# %% [markdown]
# 2 input neurons
# 
# 2 hidden neurons
# 
# 2 output neurons

# %%
x = np.array([1.5,-0.5])
# layer_1_w = np.array([
#     [0.2],
#     [0.4],
#     [-0.3],
#     [0.1]
# ])
# layer_2_w = np.array([
#     [0.6],
#     [-0.3],
#     [-0.1],
#     [-0.1]
# ])
W1 = np.array([
    [0.2, -0.3],
    [0.4, 0.1]
])

W2 = np.array([
    [0.6, -0.1],
    [-0.3, 0.4]
])

b1 = np.array([
    [0.1],
    [-0.2]
])

b2 = np.array([
    [0.1],
    [-0.1]
])

# %%
Output = ForwardPass(x, W1, b1, W2, b2)

# %%
print(Output)

# %%



