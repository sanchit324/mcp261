######################### Code from Question 1 ##############################

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Transition Probability Matrix
tpm = pd.read_excel("tpm.xlsx", header=None, sheet_name="TPM")
tpm = tpm.to_numpy()

# Cost Matrix
cost = pd.read_excel("tpm.xlsx", header=None, sheet_name="Cost Matrix")
cost = cost.iloc[7]
cost = cost.to_numpy()

# Average Time Matrix Calculation
averageTime = np.eye(18) # Identity Matrix
multiplier = tpm.copy()

# I + P + P^2 + P^3 + ... + P^59
for _ in range(59):
    averageTime +=  multiplier
    multiplier = multiplier.dot(tpm)

# Defining initial vector
initial_vector = np.zeros(18)
initial_vector[0] = 1

# Average Cost Calculation
average_cost = np.dot(cost.T, np.dot(initial_vector,averageTime))

####################### Question 2 #############################

# Setting Random Seed
np.random.seed(1234)

# Number of iterations
iterations = 1000

simulation_results = []

# Simulating over 1000 iterations
for _ in range(iterations):
    randomCost = np.random.normal(loc=cost,scale=0.1*cost)  # Random Cost
    averageCost = np.dot(randomCost.T,np.dot(initial_vector,averageTime))  # Average Cost Calculation
    simulation_results.append(averageCost) # Appending the results to list
    
mean_total_costs = np.mean(simulation_results) # Mean of Total Costs
std_dev_total_costs = np.std(simulation_results) # Standard Deviation of Total Costs

print(f"Mean Total Cost: {mean_total_costs}") # Printing Mean of Total Costs
print(f"Standard Deviation of Total Cost: {std_dev_total_costs}") # Printing Standard Deviation of Total Costs

# Plotting the histogram
plt.hist(simulation_results, bins=40)
plt.xlabel("Total Cost")
plt.ylabel("Frequency")
plt.title("Histogram of Total Cost")
plt.show()  

