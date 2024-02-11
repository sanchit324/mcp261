# Importing necessary library for permuting
from itertools import permutations

# Given distance matrix
distance_matrix = [[0, 43, 40, 33, 63],
                   [43, 0, 13, 25, 50],
                   [40, 13, 0, 14, 40],
                   [33, 25, 14, 0, 32],
                   [63, 50, 40, 32, 0]]

# Defining the initial params
num_nodes = len(distance_matrix)
best_route = None
min_distance = float('inf')
res = []

# Enumerate all possible routes
for route in permutations(range(num_nodes)):
    # Calculating the current distance
    current_distance = sum(distance_matrix[i][j] for i, j in zip(route, route[1:] + (route[0],)))
    
    # Append all the possible routes and their corresponding distances to a list
    res.append([route,current_distance])
    
    # Update the best route if the current distance is smaller and the starting route is 0
    if current_distance < min_distance and route[0] == 0:
        min_distance = current_distance
        best_route = route

# Printing all the best routes and distances from the given [routes,distance] list
for _ in range(len(res)):
    if(res[_][1] == min_distance and res[_][0][0] == 0):
        print(f"The best route is : {res[_][0]} and the corresponding distance is {res[_][1]}")
