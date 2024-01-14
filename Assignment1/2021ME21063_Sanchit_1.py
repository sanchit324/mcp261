# Import necessary libraries
import pandas as pd
import numpy as np

# Set seed to 1234
np.random.seed(1234)

# Reading the data
data = pd.read_excel('ques1_data.xlsx')

# Assigning the values of mu, sigma and gamma
mu = data.mean().values
sigma = np.cov(data.T)
gamma = 0.2

def solve(iterations):
    max_val = float('-inf')
    weights = np.empty(5)

    for _ in range(iterations):
        w = np.random.dirichlet(np.ones(5), size=1)[0]
        val = np.dot(w.T, mu) - gamma * np.dot(np.dot(w.T, sigma), w) 
        
        if max_val < val:
            max_val = val
            weights = w
    return weights, max_val

# Defining the number of iterations
iters = [10, 100, 1000, 10000, 100000, 500000]

# Storing the results in separate arrays
weights_array = []
val_array = []

for i in iters:
    weights, val = solve(i)
    weights_array.append(weights)
    val_array.append(val)

# Printing out the results
for i in range(len(iters)):
    print(f"For iterations = {iters[i]} the optimal weights are: {weights_array[i]} and the maximum value is: {val_array[i]}")

