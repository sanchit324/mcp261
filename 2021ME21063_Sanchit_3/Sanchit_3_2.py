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
        if(np.random.rand() < 0.4):
            NextRepair = Clock + 3.5
        else:
            NextRepair = Clock + 1.5
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
        if(np.random.rand() < 0.4):
            NextRepair = Clock + 3.5
        else:
            NextRepair = Clock + 1.5

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
Store_S = np.zeros((100, 1))
Store_Y = np.zeros((100, 1))

for Rep in range(100):
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
print(f'Average failure at time {SumY / 100} with average # functional components {SumS / 100}')


# Made by Sanchit ❤️ 