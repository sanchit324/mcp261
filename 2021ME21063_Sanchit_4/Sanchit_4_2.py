import numpy as np
np.random.seed(1234)

def simulate_queue(replications, simulation_time, arrival_rate, service_mean, service_std):
    wait_times = []
    length_of_stays = []
    queue_lengths = []
    server_utilizations = []

    for _ in range(replications):
        time = 0
        queue = []
        service_times = []

        while time < simulation_time:
            inter_arrival_time = np.random.exponential(1 / arrival_rate)
            time += inter_arrival_time

            # Generate service time using Gaussian distribution
            service_time = np.random.normal(service_mean, service_std)
            service_time = abs(service_time)  # Replace negative samples with their absolute values

            # Queue and service
            queue.append(time)
            service_times.append(service_time)

            while queue and queue[0] + service_times[0] <= time:
                queue.pop(0)
                service_times.pop(0)

        if len(queue) > 1:  # Check if there are wait times
            wait_time = np.mean([queue[i] - queue[i - 1] for i in range(1, len(queue))])
        else:
            wait_time = 0

        length_of_stay = np.mean([service_times[i] + wait_time for i in range(len(service_times))])
        queue_length = np.mean([len(queue) for _ in range(len(queue))])
        server_utilization = np.sum(service_times) / time if time > 0 else 0

        wait_times.append(wait_time)
        length_of_stays.append(length_of_stay)
        queue_lengths.append(queue_length)
        server_utilizations.append(server_utilization)

    avg_wait_time = np.nanmean(wait_times)
    std_wait_time = np.nanstd(wait_times)

    avg_length_of_stay = np.nanmean(length_of_stays)
    std_length_of_stay = np.nanstd(length_of_stays)

    avg_queue_length = np.mean(queue_lengths)
    std_queue_length = np.std(queue_lengths)

    avg_server_utilization = np.nanmean(server_utilizations)
    std_server_utilization = np.nanstd(server_utilizations)

    return avg_wait_time, std_wait_time, avg_length_of_stay, std_length_of_stay, \
           avg_queue_length, std_queue_length, avg_server_utilization, std_server_utilization

# Run the simulation
replications = 100
simulation_time = 500*60
arrival_rate = 8/60 
service_mean = 5
service_std = 1

result = simulate_queue(replications, simulation_time, arrival_rate, service_mean, service_std)
print("Average Wait Time:", result[0])
print("Standard Deviation of Wait Time:", result[1])
print("Average Length of Stay:", result[2])
print("Standard Deviation of Length of Stay:", result[3])
print("Average Queue Length:", result[4])
print("Standard Deviation of Queue Length:", result[5])
print("Average Server Utilization:", result[6])
print("Standard Deviation of Server Utilization:", result[7])
