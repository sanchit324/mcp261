########################### Code from part 1 ############################################

# Import necessary libraries
import pandas as pd
import numpy as np
from scipy.optimize import minimize

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
    
############################################################################################

# Defining the objective function
def objective(weights):
    return -(np.dot(weights.T, mu) - gamma * np.dot(np.dot(weights.T, sigma), weights))

# Define constraints to make sure sum of weights is 1
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

# Defining bounds to make sure the weights are non-negative
bounds = tuple((0, None) for asset in range(5))

# Starting with some equal weights
initial_weights = np.ones(5) / 5

# Optimizing the function using scipy.optimize.minimize
result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

# Optimal weights and values
optimal_weights = result.x
optimal_value = -objective(optimal_weights)

# Printing out the results
print(f"Optimal weights are: {optimal_weights} and the maximum value is: {optimal_value}")


######################### Plotting the final results###########################################

percentage_difference = np.abs(optimal_value - val_array) / optimal_value * 100

import matplotlib.pyplot as plt

plt.plot(iters, percentage_difference, marker='o')
plt.xlabel('Number of Iterations')
plt.ylabel('Absolute Percentage Difference')
plt.title('Absolute Percentage Difference vs. Number of Iterations')
plt.show()

########################## End of code ####################################################
