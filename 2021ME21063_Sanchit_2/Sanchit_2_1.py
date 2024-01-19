######################### Question 1 ##############################

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

# I + P + P^2 + P^3 + ... + P^60
for _ in range(60):
    averageTime +=  multiplier
    multiplier = multiplier.dot(tpm)

# Defining initial vector
initial_vector = np.zeros(18)
initial_vector[0] = 1

# Average Cost Calculation
average_cost = np.dot(cost.T, np.dot(initial_vector,averageTime))
print(f"Average Cost: {average_cost}")

############################## End of Question 1 ##############################