# Importing necessary libraries
import numpy as np
import pandas as pd 
import scipy.stats as st

# Defining Timer function
def Timer():
    # Defining Global Variables
    global Clock, NextFailure, NextRepair

    # Determine the next event and advance time
    if NextFailure < NextRepair:
        y = 'Failure'
        Clock = NextFailure
        NextFailure = Infinity
    else:
        y = 'Repair'
        Clock = NextRepair
        NextRepair = Infinity
    return y

# Repair Function
def Repair():
    # Defining Global Variables
    global Clock, NextFailure, NextRepair, S, Slast, Tlast, Area

    # Repair event
    # Update state and schedule future events
    S = S + 1
    if S == 1:
        NextRepair = Clock + 2.5
        NextFailure = Clock + np.ceil(6 * np.random.rand())

    # Update area under the S(t) curve
    Area = Area + Slast * (Clock - Tlast)
    Tlast = Clock
    Slast = S


# Failure Function
def Failure():
    # Defining Global Variables
    global Clock, NextFailure, NextRepair, S, Slast, Tlast, Area

    # Failure event
    # Update state and schedule future events
    S = S - 1
    if S == 1:
        NextFailure = Clock + np.ceil(6 * np.random.rand())
        NextRepair = Clock + 2.5

    # Update area under the S(t) curve
    Area = Area + Slast * (Clock - Tlast)
    Tlast = Clock
    Slast = S
    
# Defining Infinity    
Infinity = 1000000
np.random.seed(1234) # Setting seed to 1234

# Initialize the state and statistical variables
S = 2
Slast = 2
Clock = 0
Tlast = 0
Area = 0

# Schedule the initial failure event
NextFailure = np.ceil(6 * np.random.rand())
NextRepair = Infinity

# Advance time and execute events until the system fails
while not (S == 0):
    NextEvent = Timer()
    if NextEvent == 'Failure':
        Failure()
    elif NextEvent == 'Repair':
        Repair()

# Display output
# print(f'System failure at time {Clock} with average # functional components {Area / Clock}')


np.random.seed(1234) # Setting seed to 1234

# Define and initialize replication variables
SumS = 0
SumY = 0
Store_S = np.zeros((60, 1))
Store_Y = np.zeros((60, 1))

for Rep in range(60):
    # Initialize the state and statistical variables
    S = 2
    Slast = 2
    Clock = 0
    Tlast = 0
    Area = 0

    # Schedule the initial failure event
    NextFailure = np.ceil(6 * np.random.rand())
    NextRepair = Infinity

    # Advance time and execute events until the system fails
    while not (S == 0):
        NextEvent = Timer()
        if NextEvent == 'Failure':
            Failure()
        elif NextEvent == 'Repair':
            Repair()

    # Accumulate replication statistics
    SumS = SumS + Area / Clock
    SumY = SumY + Clock
    Store_S[Rep, 0] = Area / Clock
    Store_Y[Rep, 0] = Clock

# Display output
print(f'Average failure at time {SumY / 60} with average # functional components {SumS / 60}')

# Save results to Excel file using pandas
# df = pd.DataFrame({'Average # Functional Components': Store_S[:, 0], 'Time to Failure': Store_Y[:, 0]})
# df.to_excel('matresults_pandas.xlsx', index=False)


# Importing the results from MATLAB file
matlab_results = pd.read_excel('matresults.xlsx', header=None)
matlab_result = matlab_results.to_numpy().flatten()
# print(matlab_result)


# t-test to compare the results
t_value, p_value = st.ttest_ind(matlab_result, Store_S.flatten(), equal_var=True)
print(f"t_test value: {t_value}, p_test value: {p_value}")


# Made by Sanchit ❤️ 

