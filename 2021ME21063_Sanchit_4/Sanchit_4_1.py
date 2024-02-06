# Importing necessary libraries
import numpy as np
import pandas as pd 

# Defining Timer function
def Timer():
    global Clock, NextFailure, NextRepair
    if NextFailure < NextRepair[0]:
        y = 'Failure'
        Clock = NextFailure
    else:
        y = 'Repair'
        Clock = NextRepair[0]
    return y

# Repair Function
def Repair():
    global Clock, NextFailure, NextRepair, S, Slast, Tlast, Area
    S = S + 1
    del NextRepair[0]
    

# Failure Function
def Failure():
    global Clock, NextFailure, NextRepair, S, Slast, Tlast, Area
    S = S - 1
    if S > 0:
        NextFailure = Clock + np.random.choice([1,3,5,7,9])
        if NextRepair[0] == Infinity:
            NextRepair.pop(0)
        NextRepair.append(Clock + 10.5)
    
# Defining Infinity    
Infinity = 1000000
np.random.seed(1234) # Setting seed to 1234

# Define and initialize replication variables
SumY = 0
Store_Y = np.zeros((100, 1))

for Rep in range(100):
    S = 5
    Slast = 5
    Clock = 0
    Tlast = 0
    Area = 0
    NextRepair = []

    # Schedule the initial failure event
    NextFailure = np.random.choice([1,3,5,7,9])
    NextRepair.append(Infinity)

    # Advance time and execute events until the system fails
    while not (S == 0):
        NextEvent = Timer()
        if NextEvent == 'Failure':
            Failure()
        elif NextEvent == 'Repair':
            Repair()

    # Accumulate replication statistics
    SumY = SumY + Clock
    Store_Y[Rep, 0] = Clock


# Display output
print(f'Average failure at time {SumY / 100} with average standard deviaion of failure time {np.std(Store_Y)}')

average_failure_time = []

for i in range(2,8):
    SumY = 0
    Store_Y = np.zeros((100, 1))

    for Rep in range(100):
        S = i
        Slast = i
        Clock = 0
        Tlast = 0
        Area = 0
        NextRepair = []

        # Schedule the initial failure event
        NextFailure = np.random.choice([1,3,5,7,9])
        NextRepair.append(Infinity)

        # Advance time and execute events until the system fails
        while not (S == 0):
            NextEvent = Timer()
            if NextEvent == 'Failure':
                Failure()
            elif NextEvent == 'Repair':
                Repair()

        # Accumulate replication statistics
        SumY = SumY + Clock
        Store_Y[Rep, 0] = Clock
        
    average_failure_time.append(SumY/100)
        
        
import matplotlib.pyplot as plt
num_machines = [2,3,4,5,6,7]
plt.figure(figsize=(8, 5))
plt.plot(num_machines,average_failure_time)
plt.xlabel('Number of Machines')
plt.ylabel('Time to Failure (days)')
plt.title('Average Time to Failure vs. Number of Machines')
plt.grid(True)
plt.show()