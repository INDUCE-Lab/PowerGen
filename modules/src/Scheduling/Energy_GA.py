
'''
Title:        PowerGen Toolkit
Description:  PowerGen (Power Generation Dataset) Toolkit for Generating Resources Utlization and Corresponding Power Consumption in Edge and Cloud Computing Data Centers
Licence:      GPL - http://www.gnu.org/copyleft/gpl.html

If you are using any ideas, algorithms, packages, codes, datasets, workload, results, and plots, included in the power-modeling directory please cite
the following paper:

https://doi.org/TBD">Leila Ismail, and Huned Materwala, "PowerGen: Resources Utilization
and Power Consumption Data Generation Framework for Energy Prediction in Edge and Cloud Computing",
ANT 2024

'''

import numpy as np
import random
import sys
import time
import datetime
import csv
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


np.random.seed(42) #specifying seed for random number generation


#########################################################################################
#########################################################################################

"""
Variables and parameters
"""

Vehicles = 100

Minimum_CPU = 10
Maximum_CPU = 90
Num_requests = Vehicles
Arrival_rate = 2.0

Cloud_servers = 5
Edge_servers = 10

Vehicle_RSU_bandwidth = 125000
Minimum_edgeCloudBW = 125000
Maximum_edgeCloudBW = 250000

Minimum_Xlocation = 0
Maximum_Xlocation = 1000
Minimum_Ylocation = 0
Maximum_Ylocation = 1000

swap_factor = 0.05

latency = 0.1
processing = 0.9
deadline = latency + processing
latency_req = [latency] * Num_requests #Array to store latency requirements of the requests
processing_req = [processing] * Num_requests #Array to store latency requirements of the requests
deadline_req = [deadline] * Num_requests #Array to store latency requirements of the requests


'''
Below is the initialization of the genetic
algorithms parameters used in the experiments
'''
random.seed(10)
population_size = 2*Num_requests
cross_rate, mutation_rate  = 0.95, 0.01
Generations = 100
loop = 1


#########################################################################################
#########################################################################################

"""
This function simulates 'Num_requests' requests, where each request represent an application request from a vehicle.  Each request consists of CPU
utilization (%), length (in Million Instructions), and size (in terms of Megabits).  Requests are generated based on poison distribution with
exponential interarrival times.
"""


"""
Generating CPU utilization for each task between minimum (min_CPU) and maximum (max_CPU) CPU utilization values.
Each CPU utilization value is generated randomly from a Uniform distribution in an interval between Minimum_CPU and Maximum_CPU.
"""
CPU_util_task = np.random.uniform(Minimum_CPU,Maximum_CPU,Num_requests) #generating CPU utilization for 'Num_requests' tasks uniformly between min_CPU and max_CPU

low_CPU = []
up_CPU = []
## Calculating upper and lower bound cpu util
for k in range(0, Num_requests):
    low_CPU.append(math.floor(CPU_util_task[0]/10))
    up_CPU.append(math.ceil(CPU_util_task[0]/10))

"""
Generating length (in Million Instructions) and size (in Megabits) of each request. The minimum and maximum values for request length
and size are based on different vehicular applications such as face recognition and object detection for autonomous driving,
augmented reality, VANET-based health monitoring, and infotainment.  These values are retrieved from the following references:
> N. Auluck, A. Azim, K. Fizza, Improving the schedulability of real-time tasks using fog computing, IEEE Trans. Serv. Comput. (2019)
> A. Jaddoa, G. Sakellari, E. Panaousis, G. Loukas, P.G. Sarigiannidis, Dynamic decision support for resource offloading in heterogeneous
Internet of Things environments, Simul. Model. Pract. Theory 101 (2020) 102019.
> J. Almutairi, M. Aldossary, A novel approach for IoT tasks offloading in edge-cloud environments, J. Cloud Comput. 10 (1) (2021) 1–19.
"""

length_task = np.random.randint(500, 5000+1, Num_requests)  #generating length of each request between 500 and 5000 Million Instructions
size_task = np.random.uniform(0,8,Num_requests) #generating size of each request uniformly between Minimum_size and Maximum_size


"""
Generating interarrival times between each request using exponential distribution.
"""
IAT = [0] * Num_requests #initializing an array to store interarrival time between requests
IAT[0] = 0 #no interarrival time for the first request 
for r in range (1,Num_requests):
    IAT[r] = np.random.exponential(scale=1/Arrival_rate)   #generating exponential interarrval time with arrival rate of 'Arrival_rate' requests per second


#########################################################################################
#########################################################################################

"""
This function simulates the network for IoV infrastructure with integrated edge-cloud computing system.  In particular, it simulates data transfer
bandwidth between vehicle - Roadside Units and between Roadside Units - cloud servers. A constant bandwidth is considered throughout the simulation
in this version of the toolkit.  Bandwidth between each vehicle and Roadside Unit is same and that between each Roadside Unit and cloud server is
different.
"""

vehicle_RSU_bw = Vehicle_RSU_bandwidth #generating a constant bandwidth of 'Vehicle_RSU_bandwidth' Megabits/second between each vehicle and connected Roadside Unit
RSU_cloud_bw = np.random.randint(Minimum_edgeCloudBW, Maximum_edgeCloudBW, size=(Edge_servers))    #generating random Roadside unit - cloud bandwidth between Minimum_edgeCloudBW and Maximum_edgeCloudBW Megabits for each RSU and cloud server


#########################################################################################
#########################################################################################

"""
This function simulates the RoadSide Units.  Each vehicle communicates to a RoadSide Unit and each Roadside Unit is equipped with an edge server.
Consequently, the number of RoadSide Units are edge servers are equal in the simulated network.  RoadSide Units are placed equidistant.
"""

RSU_position_X = [0] * Edge_servers #initializing an array to store X locations of the simulated Roadside Units
RSU_position_Y = [0] * Edge_servers #initializing an array to store Y locations of the simulated Roadside Units

RSU_coverage_X = [[0 for x in range(2)] for y in range(Edge_servers)]   #initializing a matrix to store X locations for the lower and upper coverage area locations of the simulated Roadside Units
RSU_coverage_Y = [[0 for x in range(2)] for y in range(Edge_servers)]   #%initializing an array to store Y locations for the lower and upper coverage area locations of the simulated Roadside Units

for i in range(Edge_servers):  #generating the location of Roadside Units

    """
    To calculate the X location for each RoadSide Unit, the simulated area is first divided in 'Num_edge_servers' parts.  Then an
    RoadSide Unit is placed at the center of each part. To calculate the Y location for each RoadSide Unit, the vertical distance
    of the simulated area is divided into half. Consequenlty, each RSU is placed in the center of a part in the simulated area.
    """

    RSU_position_X[i] = (i+1)*(((Maximum_Xlocation)-(Minimum_Xlocation))/Edge_servers) - ((((Maximum_Xlocation)-(Minimum_Xlocation))/Edge_servers)/2)
    RSU_position_Y[i] = ((Maximum_Ylocation)-(Minimum_Ylocation))/2

    RSU_coverage_X[i][0] = (i)*(((Maximum_Xlocation)-(Minimum_Xlocation))/Edge_servers)
    RSU_coverage_X[i][1] = (i+1)*(((Maximum_Xlocation)-(Minimum_Xlocation))/Edge_servers)
    RSU_coverage_Y[i][0] = Minimum_Ylocation
    RSU_coverage_Y[i][1] = Maximum_Ylocation


#########################################################################################
#########################################################################################

"""
This function simulates 'Num_edge_servers' edge servers, where the variable 'Num_edge_servers' is user-defined. Each edge server is
created with certain processing speed and a power consumption profile.  The processing speed is defined in terms of Million Instructions
Per Second (MIPS). Power consumption profile consists on server's CPU utilization from 0% - 100% at an interval of 10% and
the corresponding power consumption.

It will create 'n_types_edge' types of heterogeneous edge servers (i.e., servers with different processing speed and power consumption profile).
The function should be modified to change server specifications and/or to add more server types. In this code the value of 'n_types_edge' is 3,
i.e., three different edge server types are created.

While simulating the IoV network, if Num_edge_servers > n_types_edge, then each server type will be created for
Num_edge_servers/n_types_edge times.
"""


""" Defining processing speed of 'n_types_edge' (3 in this case) edge server types """
processing_speed_edge = [1300, 2500, 1700]


""" Defining memory of 'n_types_edge' (3 in this case) edge server types """
memory_edge = [2048, 2048, 2048]

""" Defining power consumption profile of 'n_types_edge' (3 in this case) edge server types """
power_cons_edge = np.array([[138.2685, 142.2829, 146.7379, 151.1492, 155.3824, 159.9734, 164.4558, 169.1667, 173.8268, 178.4852, 181.7913],     #energy profile for type 1 cloud server
                [54.1, 78.4, 88.5, 99.5, 115, 126, 143, 165, 196, 226, 243],                                                                #energy profile for type 2 cloud server
                [204.2420, 204.9672, 205.9185, 206.6314, 207.5923, 208.5179, 209.1885, 210.2377, 211.1731, 211.8091, 214.9755]])             #energy profile for type 3 cloud server
                
"""

Allocating procesing speed to each edge server. If the number of edge servers are more than the types of edge servers, then same
types are assigned to multiple servers. For instance, edge server 1 will be assigned with speed 1, edge server 2 -> speed 2,
edge server 3 -> speed 3, then again edge server 4 -> speed 1, edge server 5 -> speed 2, edge server 6 -> speed 3,
edge server 7 -> speed 1 and so on.

""" 

"""initializing the value of edge server type with 1"""
n_types_edge = 1

"""initializing an empty array to store the values of edge servers' processing capabolities"""
edge_server_speed = [0] * Edge_servers

for s in range(Edge_servers):  #looping over the total number of edge servers
    if n_types_edge > len(processing_speed_edge): #checking if the current value of edge server type is greater than total edge server types
        n_types_edge = 1   #reinitialzing the value of edge server type with 1 if current value of edge sever type is greater than total edge server types
    edge_server_speed[s] = processing_speed_edge[n_types_edge-1]    #assigning the processing speed of edge servers
    n_types_edge += 1  #incrementing the current value of edge server type

"""

Allocating memory to each edge server. If the number of edge servers are more than the types of edge servers, then same
types are assigned to multiple servers. For instance, edge server 1 will be assigned with memory 1, edge server 2 -> memory 2,
edge server 3 -> memory 3, then again edge server 4 -> memory 1, edge server 5 -> memroy 2, edge server 6 -> memory 3,
edge server 7 -> memory 1 and so on.

""" 

"""initializing the value of edge server type with 1"""
n_types_edge = 1

"""initializing an empty array to store the values of edge servers' processing capabolities"""
edge_server_memory = [0] * Edge_servers

for s in range(Edge_servers):  #looping over the total number of edge servers
    if n_types_edge > len(memory_edge): #checking if the current value of edge server type is greater than total edge server types
        n_types_edge = 1   #reinitialzing the value of edge server type with 1 if current value of edge sever type is greater than total edge server types
    edge_server_memory[s] = memory_edge[n_types_edge-1]    #assigning the processing speed of edge servers
    n_types_edge += 1  #incrementing the current value of edge server type


"""

Allocating power consumption profile to each edge server. If the number of edge servers are more than the types of edge servers,
then same power consumption profile is assigned to multiple servers. For instance, edge server 1 will be assigned with power
consumption profile 1, edge server 2 -> power consumption profile 2, edge server 3 -> power consumption profile 3,
then again edge server 4 -> power consumption profile 1, edge server 5 -> power consumption profile 2, edge server 6 -> power consumption
profile 3, edge server 7 -> power consumption profile 1 and so on.

"""

"""initializing the value of edge server type with 1"""
n_types_edge = 1

"""initializing an empty array to store the values of edge servers' power consumption"""
edge_server_power = [[0 for x in range(len(power_cons_edge[0]))] for y in range(Edge_servers)]

for p in range(Edge_servers):  #looping over the total number of edge servers
    if n_types_edge > len(power_cons_edge):    #checking if the current value of edge server type is greater than total edge server types
        n_types_edge = 1   #reinitialzing the value of edge server type with 1 if current value of edge sever type is greater than total edge server types
    edge_server_power[p] = power_cons_edge[n_types_edge-1,:]    #assigning the processing speed of edge servers
    n_types_edge += 1  #incrementing the current value of edge server type


#########################################################################################
#########################################################################################

"""
This function simulates 'Num_cloud_server' cloud servers, where the variable 'Num_cloud_server' is user-defined. Each cloud server is
created with certain processing speed and a power consumption profile.  The processing speed is defined in terms of Million Instructions
Per Second (MIPS). Power consumption profile consists on server's CPU utilization from 0% - 100% at an interval of 10% and
the corresponding power consumption.

It will create 'n_types_cloud' types of heterogeneous cloud servers (i.e., servers with different processing speed and power consumption profile).
The function should be modified to change server specifications and/or to add more server types. In this code the value of 'n_types_cloud' is 3,
i.e., three different cloud server types are created.

While simulating the IoV network, if Num_cloud_server > n_types_cloud, then each server type will be created for
Num_cloud_server/n_types_cloud times.
"""


""" Defining processing speed of 'n_types_cloud' (3 in this case) cloud server types """
processing_speed_cloud = [2750, 3000, 6000]


""" Defining memory of 'n_types_cloud' (3 in this case) edge server types """
memory_cloud = [4096, 4096, 4096]


""" Defining power consumption profile of 'n_types_cloud' (3 in this case) cloud server types """
power_cons_cloud = np.array([[265, 531, 624, 718, 825, 943, 1060, 1158, 1239, 1316, 1387],  #energy profile for type 1 cloud server
                [45, 83.7, 101, 118, 133, 145, 162, 188, 218, 248, 276],                #energy profile for type 2 cloud server
                [127, 220, 254, 293, 339, 386, 428, 463, 497, 530, 559]])               #energy profile for type 3 cloud server

 
"""

Allocating procesing speed to each cloud server. If the number of cloud servers are more than the types of cloud servers, then same
types are assigned to multiple servers. For instance, cloud server 1 will be assigned with speed 1, cloud server 2 -> speed 2,
cloud server 3 -> speed 3, then again cloud server 4 -> speed 1, cloud server 5 -> speed 2, cloud server 6 -> speed 3,
cloud server 7 -> speed 1 and so on.


"""

"""initializing the value of cloud server type with 1"""
n_types_cloud = 1

"""initializing an empty array to store the values of cloud servers' processing capabolities"""
cloud_server_memory = [0] * Cloud_servers

for s in range(Cloud_servers):  #looping over the total number of cloud servers
    if n_types_cloud > len(memory_cloud): #checking if the current value of cloud server type is greater than total cloud server types
        n_types_cloud = 1   #reinitialzing the value of cloud server type with 1 if current value of cloud sever type is greater than total cloud server types
    cloud_server_memory[s] = memory_cloud[n_types_cloud-1]    #assigning the processing speed of cloud servers
    n_types_cloud += 1  #incrementing the current value of cloud server type


"""

Allocating memory to each cloud server. If the number of cloud servers are more than the types of cloud servers, then same
types are assigned to multiple servers. For instance, cloud server 1 will be assigned with memory 1, cloud server 2 -> memory 2,
cloud server 3 -> memory 3, then again cloud server 4 -> memory 1, cloud server 5 -> memory 2, cloud server 6 -> memory 3,
cloud server 7 -> memory 1 and so on.##for r in range(Num_requests):
##    list_possible_servers = []


"""

"""initializing the value of cloud server type with 1"""
n_types_cloud = 1

"""initializing an empty array to store the values of cloud servers' processing capabolities"""
cloud_server_speed = [0] * Cloud_servers

for s in range(Cloud_servers):  #looping over the total number of cloud servers
    if n_types_cloud > len(processing_speed_cloud): #checking if the current value of cloud server type is greater than total cloud server types
        n_types_cloud = 1   #reinitialzing the value of cloud server type with 1 if current value of cloud sever type is greater than total cloud server types
    cloud_server_speed[s] = processing_speed_cloud[n_types_cloud-1]    #assigning the processing speed of cloud servers
    n_types_cloud += 1  #incrementing the current value of cloud server type


"""

Allocating power consumption profile to each cloud server. If the number of cloud servers are more than the types of cloud servers,
then same power consumption profile is assigned to multiple servers. For instance, cloud server 1 will be assigned with power
consumption profile 1, cloud server 2 -> power consumption profile 2, cloud server 3 -> power consumption profile 3,
then again cloud server 4 -> power consumption profile 1, cloud server 5 -> power consumption profile 2, cloud server 6 -> power consumption
profile 3, cloud server 7 -> power consumption profile 1 and so on.

"""

"""initializing the value of cloud server type with 1"""
n_types_cloud = 1

"""initializing an empty array to store the values of cloud servers' power consumption"""
cloud_server_power = [[0 for x in range(len(power_cons_cloud[0]))] for y in range(Cloud_servers)]

for p in range(Cloud_servers):  #looping over the total number of cloud servers
    if n_types_cloud > len(power_cons_cloud):    #checking if the current value of cloud server type is greater than total cloud server types
        n_types_cloud = 1   #reinitialzing the value of cloud server type with 1 if current value of cloud sever type is greater than total cloud server types
    cloud_server_power[p] = power_cons_cloud[n_types_cloud-1,:]    #assigning the processing speed of cloud servers
    n_types_cloud += 1  #incrementing the current value of cloud server type


#########################################################################################
#########################################################################################

"""
This function simulates the vechiles.  In particular it simulates the position of each vehicle and determines the communicating
RoadSide Unit for each vehcile.  This is by determining under which RoadSide Unit's coverage area each vehicle is located.
"""

"""
Generating X and Y locations for each vehicle randomly within the simulated area.  The simulated area is determined by
the extreme (minimum and maximun) X and Y locations of the simulated area.
"""

Vehicle_Xlocation= np.random.randint(Minimum_Xlocation, Maximum_Xlocation+1, Vehicles)
Vehicle_Ylocation= np.random.randint(Minimum_Ylocation, Maximum_Ylocation+1, Vehicles)

"""
Finding the communicating RoadSide Unit for each vehicle.  This is by determining under which RoadSide Units X-axis coverage
range does the X_location of each vehicle lies.  The Y covergae range and Y_location for RoadSide Units and vehicles respectively
are not considered as all the RoadSide Units are placed in a line horizontally.  If the RoadSide Units are placed as a grid,
then Y covergae range and Y_location for RoadSide Units and vehicles respectively should be considered.
"""

Connected_RSU = [0] * Vehicles #initializing an array to store communicating RoadSide Unit for each vehicle

for v in range(Vehicles):
    for r in range(Edge_servers):
        """
        Finding in which RoadSide Unit's X_coverage range does the x_location of each vehicle lies.  If the coverage range is
        found, then that RoadSide Unit is saved as connected RSU for that vehicle.
        """

        if Vehicle_Xlocation[v] > RSU_coverage_X[r][0] and Vehicle_Xlocation[v] < RSU_coverage_X[r][1]:
            break

    Connected_RSU[v] = r

"""
Generating a random number 0 or 1 for each vehicle to determine whether the vehicle will be in range of RSU while
receiving request or not.  1 means in the range and 0 means outside the range.
"""
Vehicle_position= np.random.randint(0, 2, Vehicles)

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

'''Compute the values of Tmax, ViolationsMax, and Emax'''

'''To compute the value of Tmax, all requests are processed on edge servers.
Each request is executed by the edge server to which it is submitted.'''

server_alloc_max = [0] * Num_requests #initializing an array to store server allocation of the requests
comm_time_max = [0] * Num_requests #initializing an array to store communication time of the requests
wait_time_max = [0] * Num_requests #initializing an array to store waiting time of the requests
proc_time_max = [0] * Num_requests #initializing an array to store processing time of the requests
io_time_max = [0] * Num_requests #initializing an array to store i/o time of the requests
exec_time_max = [0] * Num_requests #initializing an array to store execution time of the requests


Tasks_in_edge = {}
for server in range(0, Edge_servers):
    Tasks_in_edge[server] = []
for request in range(0, Num_requests):
    Tasks_in_edge[Connected_RSU[request]].append(request)
for server in range(0, Edge_servers):
    if len(Tasks_in_edge[server]) != 0:
        sum_wait = 0
        wait_time_temp = 0
        for request in Tasks_in_edge[server]:
            server_alloc_max[request] = server+1
            comm_time_max[request] = (size_task[request]/vehicle_RSU_bw +
                                  (0.01*size_task[r])/vehicle_RSU_bw)
            
            wait_time_max[request] = wait_time_temp
            running_time = length_task[request]/edge_server_speed[server]
            proc_time_temp = running_time + wait_time_temp
            wait_time_temp = proc_time_temp
            sum_wait = wait_time_temp + sum_wait
            proc_time_max[request] = proc_time_temp

            if math.ceil(size_task[request]/edge_server_memory[server]) == 0:
                io_time_max[request] = 0
            else:
                io_time_max[request] = swap_factor*((math.ceil(size_task[request]/edge_server_memory[server]))-1)

            exec_time_max[request] = (comm_time_max[request] + proc_time_max[request] + io_time_max[request])

Tmax = np.sum(exec_time_max)


'''To compute the maximum SLA violations'''

Tasks = list(range(1,Num_requests+1))
latency_violations_max = []
processing_violations_max = []
deadline_violations_max = []

for i in range(0, Num_requests):
    if comm_time_max[i] > latency:
        latency_violations_max.append(comm_time_max[i] - latency_req[i])
    else:
        latency_violations_max.append(0)
        
    if proc_time_max[i] > processing:
        processing_violations_max.append(proc_time_max[i] - processing_req[i])
    else:
        processing_violations_max.append(0)
        
    if exec_time_max[i] > deadline:
        deadline_violations_max.append(exec_time_max[i] - deadline_req[i])
    else:
        deadline_violations_max.append(0)

max_latency_violations = np.sum(latency_violations_max)
max_processing_violations = np.sum(processing_violations_max)
max_deadline_violations = np.sum(deadline_violations_max)


'''To compute the value of Emax, all requests are processed on a server with the highest Pmax value'''

power_consp_max = [0] * Num_requests #initializing an array to store power consumption of the requests
energy_consp_max = [0] * Num_requests #initializing an array to store energy consumption of the requests

for request in range(Num_requests):
    power_consp_max[request] = (cloud_server_power[0][low_CPU[request]] +
                                        ((cloud_server_power[0][low_CPU[request]] - cloud_server_power[0][up_CPU[request]])*
                                         (CPU_util_task[request] - low_CPU[request]*10)/10))

    proc_time = length_task[request]/cloud_server_speed[0]
    if math.ceil(size_task[request]/cloud_server_memory[0]) == 0:
        io_time = 0
    else:
        io_time = swap_factor*((math.ceil(size_task[request]/cloud_server_memory[0]))-1)
    
    energy_consp_max[request] = power_consp_max[request]*(proc_time + io_time)

Emax = np.sum(energy_consp_max)

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

'''
Below is the code for randomly generating
initial population.  For each request, a server
is randomly selected from a set of servers that
include the edge server to which the vehicle
submits the request and all the cloud servers
'''

def population_initialization(population_size,
                              Num_requests,
                              Connected_RSU,
                              Edge_servers, Cloud_servers):
    random.seed(10)
    Initial_pop = np.zeros((population_size, Num_requests), dtype = int)
    for i in range(population_size):
        for j in range(Num_requests):
            possible_servers = [Connected_RSU[j],*list(range(Edge_servers, Edge_servers+Cloud_servers))]
            Initial_pop[i][j] = random.choice(possible_servers)

    return Initial_pop


#########################################################################################
#########################################################################################


'''
First, we compute the communication time for each request is 
computed.  To compute the communication time, first it
is determined whether the request is scheduled to
edge or cloud server.  If the request is scheduled to
an edge, then the communication time is calculated based on
whether or not the vehicle will be in the range of RSU
while receiving the respone.  If the request is scheduled to
a cloud server, the communication time will be based on the RSU
where the vehicle will be while receiving the response.

Second, the processing time is calculated.  The waiting time is
added as one by one requests are processed on the servers

Third the power and energy consumption are computed.

Fourth the violations are calculated.

The adaptive fitness is then calculated
'''

def indices(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

def fitness_evaluation(population_size,
                       Num_requests,
                       population,
                       Edge_servers,
                       Cloud_servers,
                       Vehicle_position,
                       size_task,
                       length_task,
                       vehicle_RSU_bw,
                       RSU_cloud_bw,
                       Connected_RSU,
                       edge_server_memory,
                       cloud_server_memory,
                       edge_server_speed,
                       cloud_server_speed,
                       edge_server_power,
                       cloud_server_power,
                       swap_factor,
                       CPU_util_task,
                       Emax):
        
    comm_time = np.zeros((population_size, Num_requests))
    wait_time = np.zeros((population_size, Num_requests))
    proc_time = np.zeros((population_size, Num_requests))
    io_time = np.zeros((population_size, Num_requests))
    exec_time = np.zeros((population_size, Num_requests))
    sum_exec_time = np.zeros((population_size, 1))
    power_consp = np.zeros((population_size, Num_requests))
    energy_consp = np.zeros((population_size, Num_requests))
    latency_violations = np.zeros((population_size, Num_requests))
    processing_violations = np.zeros((population_size, Num_requests))
    deadline_violations = np.zeros((population_size, Num_requests))
    sum_energy = np.zeros((population_size, 1))
    fitness = np.zeros((population_size, 1))
    objective = np.zeros((population_size, 1))
    norm_violations = np.zeros((population_size, 1))
    decision_time = np.zeros((Generations,1))
    ##Violations later

    for pop in range(population_size):

        Tasks_in_server = {}
        for server in range(0, Edge_servers+Cloud_servers):
            Tasks_in_server[server] = []


        for request in range(0, Num_requests):
            Tasks_in_server[population[pop][request]].append(request)

        for server in range(0, Edge_servers+Cloud_servers):
            if len(Tasks_in_server[server]) != 0:
                if server < Edge_servers:
                    sum_wait = 0
                    wait_time_temp = 0
                    ruuning_time_temp = 0

                    for request in Tasks_in_server[server]:
                        if Vehicle_position[request] == 0:
                            comm_time[pop][request] = (size_task[request]/vehicle_RSU_bw +
                                                       (0.01*size_task[r])/vehicle_RSU_bw)


                        elif Vehicle_position[request] == 1:
                            comm_time[pop][request] = (size_task[request]/vehicle_RSU_bw + (0.01*size_task[r])/RSU_cloud_bw[Connected_RSU[request]] +
                                          (0.01*size_task[request])/RSU_cloud_bw[random.choice(list(range(0,Edge_servers)))] +
                                          (0.01*size_task[request])/vehicle_RSU_bw)

                        wait_time[pop][request] = wait_time_temp
                        ruuning_time_temp = length_task[request]/edge_server_speed[server]
                        proc_time_temp = ruuning_time_temp + wait_time_temp
                        wait_time_temp = proc_time_temp
                        sum_wait = wait_time_temp + sum_wait
                        proc_time[pop][request] = proc_time_temp

                        if math.ceil(size_task[request]/edge_server_memory[server]) == 0:
                            io_time[pop][request] = 0

                        else:
                            io_time[pop][request] = swap_factor*((math.ceil(size_task[request]/edge_server_memory[server]))-1)

                        exec_time[pop][request] = (comm_time[pop][request]+
                                                   proc_time[pop][request]+
                                                   io_time[pop][request])

                        power_consp[pop][request] = (edge_server_power[server][low_CPU[request]] +
                                        ((edge_server_power[server][low_CPU[request]] - edge_server_power[server][up_CPU[request]])*
                                         (CPU_util_task[request] - low_CPU[request]*10)/10))

                        energy_consp[pop][request] = power_consp[pop][request]*(proc_time[pop][request]+io_time[pop][request])


                else:
                    sum_wait = 0
                    wait_time_temp = 0
                    ruuning_time_temp = 0

                    for request in Tasks_in_server[server]:
                        if Vehicle_position[request] == 0:
                            comm_time[pop][request] = (size_task[request]/vehicle_RSU_bw +
                                                  size_task[request]/RSU_cloud_bw[Connected_RSU[request]] +
                                                  (0.01*size_task[request])/RSU_cloud_bw[Connected_RSU[request]] +
                                                  (0.01*size_task[request])/vehicle_RSU_bw)
                        elif Vehicle_position[request] == 1:
                            comm_time[pop][request] = (size_task[request]/vehicle_RSU_bw +
                                                       size_task[request]/RSU_cloud_bw[Connected_RSU[request]] +
                                                       (0.01*size_task[request])/RSU_cloud_bw[random.choice(list(range(0,Edge_servers)))] +
                                                       (0.01*size_task[request])/vehicle_RSU_bw)

                        wait_time[pop][request] = wait_time_temp
                        ruuning_time_temp = length_task[request]/cloud_server_speed[server-Edge_servers]
                        proc_time_temp = ruuning_time_temp + wait_time_temp
                        wait_time_temp = proc_time_temp
                        sum_wait = wait_time_temp + sum_wait
                        proc_time[pop][request] = proc_time_temp

                        if math.ceil(size_task[request]/cloud_server_memory[server-Edge_servers]) == 0:
                            io_time[pop][request] = 0
                        else:
                            io_time[pop][request] = swap_factor*((math.ceil(size_task[request]/cloud_server_memory[server-Edge_servers]))-1)


                        power_consp[pop][request] = (cloud_server_power[server-Edge_servers][low_CPU[request]] +
                                        ((cloud_server_power[server-Edge_servers][low_CPU[request]] - cloud_server_power[server-Edge_servers][up_CPU[request]])*
                                         (CPU_util_task[request] - low_CPU[request]*10)/10))

                        energy_consp[pop][request] = power_consp[pop][request]*(proc_time[pop][request]+io_time[pop][request])


                        exec_time[pop][request] = (comm_time[pop][request]+
                                              proc_time[pop][request]+
                                              io_time[pop][request])

        '''Computing the summation of total execution times and total
        energy consumption for all the requests in a solution
        '''
        sum_exec_time[pop] = np.sum(exec_time[pop])
        sum_energy[pop] = np.sum(energy_consp[pop])

        '''Computing the latency, processing, and deadline violations'''
        for request in range(0, Num_requests):
            if comm_time[pop][request] > latency:
                latency_violations[pop][request] = comm_time[pop][request] - latency

            if proc_time[pop][request] > processing:
                processing_violations[pop][request] = proc_time[pop][request] - processing

            if exec_time[pop][request] > deadline:
                deadline_violations[pop][request] = exec_time[pop][request] - deadline

        '''Computing normalized violations'''
        def maximumSum(list1):
            return(sum(max(list1, key = sum)))


        if maximumSum(latency_violations) == 0:
            norm_lat_violations = 0
        else:
            norm_lat_violations = np.sum(latency_violations[pop])/max_latency_violations

        if maximumSum(processing_violations) == 0:
            norm_proc_violations = 0
        else:
            norm_proc_violations = np.sum(processing_violations[pop])/max_processing_violations

        if maximumSum(deadline_violations) == 0:
            norm_deadline_violations = 0
        else:
            norm_deadline_violations = np.sum(deadline_violations[pop])/max_deadline_violations

        norm_violations[pop] = (1/3)*(norm_lat_violations+norm_proc_violations+norm_deadline_violations)


    #norm_sum_exec_time = normalize(sum_exec_time)
    #norm_sum_energy = normalize(sum_energy)

    norm_sum_energy = sum_energy/Emax

    
    for i in range(population_size):
        objective[i] = norm_sum_energy[i]


    '''Calculating  fitness'''
    for i in range(population_size):
        fitness[i] = 1/objective[i]
   
    return objective,fitness,sum_energy,sum_exec_time,latency_violations,processing_violations,deadline_violations,energy_consp

#########################################################################################
#########################################################################################

'''
Below is the code for Roulette wheel selection
'''

def selection(Num_requests, population_size, population, fitness):
    sum_fitness = float(sum(fitness))
    probabilities = [f /sum_fitness for f in fitness]
    cumm_probabilities = [sum(probabilities[:i+1])
                          for i in range(len(probabilities))]

    selected_solution = []
    for i in range(population_size):
        r = random.random()
        value = next(x for x, val in enumerate(cumm_probabilities)
                                  if val >r)
        if value > 0:
            selection = value - 1
        else:
            selection = value
        selected_solution.append(selection)

    pop_after_selection = np.zeros((population_size, Num_requests), dtype = int)
    for i in range(population_size):
        pop_after_selection[i] = population[selected_solution[i]] 
       
    return pop_after_selection


#########################################################################################
#########################################################################################

'''
Below is the code for single point crossover
'''

def crossover(population_size,population,cross_rate,
              Num_requests,Edge_servers,Cloud_servers,
              Vehicle_position,size_task,length_task,
              Vehicle_RSU_bandwidth,RSU_cloud_bw,
              Connected_RSU,edge_server_memory,cloud_server_memory,
              edge_server_speed,cloud_server_speed,edge_server_power,
              cloud_server_power,swap_factor,CPU_util_task,Emax):

    
    selection_sol_crossover = []
    for i in range(population_size):
        r = random.random()
        if r < cross_rate:
            selection_sol_crossover.append(i)

    parent_pairs = np.zeros((math.ceil(len(selection_sol_crossover)/2), 2), dtype = int)
    if len(selection_sol_crossover) % 2 != 0:
        additional_solution = random.choice(selection_sol_crossover)
        selection_sol_crossover.append(additional_solution)

    index = 0
    while len(selection_sol_crossover) != 0:
        parent_pairs[index][0], parent_pairs[index][1] = random.sample(selection_sol_crossover,2)
        [selection_sol_crossover.remove(parent_pairs[index][i]) for i in range(2)]
        index = index+1

    for i in range(len(parent_pairs)):
        temp_pop = np.zeros((4, Num_requests), dtype = int)
        parent_1 = population[parent_pairs[i][0]]
        parent_2 = population[parent_pairs[i][1]]
        child_1, child_2 = parent_1.copy(), parent_2.copy()
        cutoff = random.randint(1,Num_requests-1)
        child_1[:cutoff] = parent_1[:cutoff]
        child_2[cutoff:] = parent_2[cutoff:]
        child_2[:cutoff] = parent_2[:cutoff]
        child_2[cutoff:] = parent_1[cutoff:]

        temp_pop[0] = parent_1
        temp_pop[1] = parent_2
        temp_pop[2] = child_1
        temp_pop[3] = child_2

        objective,fitness,energy,time,latency_violations,processing_violations,deadline_violations,Energy_request = fitness_evaluation(4, Num_requests, temp_pop,Edge_servers,
                                     Cloud_servers, Vehicle_position, size_task,length_task,
                                     vehicle_RSU_bw,RSU_cloud_bw, Connected_RSU,edge_server_memory,
                                     cloud_server_memory, edge_server_speed,cloud_server_speed,
                                     edge_server_power,cloud_server_power,swap_factor,CPU_util_task,Emax)            

        fitness = [item for sublist in fitness for item in sublist]

        max_fitness = max(fitness)
        second_max = max(fitness, key=lambda x: min(fitness)-1 if (x == max_fitness) else x)
        index_max = fitness.index(max_fitness)
        index_second_max = fitness.index(second_max)
        if index_max == 0:
            population[parent_pairs[i][0]] = parent_1
        elif index_max == 1:
            population[parent_pairs[i][0]] = parent_2
        elif index_max == 2:
            population[parent_pairs[i][0]] = child_1
        elif index_max == 3:
            population[parent_pairs[i][0]] = child_2

        if index_second_max == 0:
            population[parent_pairs[i][1]] = parent_1
        elif index_max == 1:
            population[parent_pairs[i][1]] = parent_2
        elif index_max == 2:
            population[parent_pairs[i][1]] = child_1
        elif index_max == 3:
            population[parent_pairs[i][1]] = child_2

    return population

#########################################################################################
#########################################################################################

def mutation(Num_requests, population_size,
             mutation_rate, population,Connected_RSU,
             Edge_servers, Cloud_servers):
    
    total_genes = Num_requests*population_size
    num_mutation = math.ceil(total_genes*mutation_rate)

    for i in range(num_mutation):
        r = random.randint(1, total_genes)
        rem = r % Num_requests
        if rem == 0:
            solution = ((r-rem)//Num_requests)-1
            possible_mutations = [Connected_RSU[Num_requests-1],*list(range(Edge_servers, Edge_servers+Cloud_servers))]
            population[solution][Num_requests-1] = random.choice(possible_mutations)
        else:
            solution = ((r-rem)//Num_requests)
            possible_mutations = [Connected_RSU[rem],*list(range(Edge_servers, Edge_servers+Cloud_servers))]
            population[solution][rem-1] = random.choice(possible_mutations)

    return population


#########################################################################################
#########################################################################################

'''Genetic algorithm main program'''

best_fitness_per_generation = []
best_objective_per_generation = []
best_energy_per_generation = []
best_time_per_generation = []

latency_per_generation = []
processing_per_generation = []
deadline_per_generation = [] 

total_latency_per_generation = []
total_processing_per_generation = []
total_deadline_per_generation = []
    
decision_time_per_generation = []


start_time_genetic = datetime.datetime.now()


Initial_population = population_initialization(population_size,
                                               Num_requests,
                                               Connected_RSU,
                                               Edge_servers,
                                               Cloud_servers)

objective,fitness,energy,time,latency_violations,processing_violations,deadline_violations,Energy_request =  fitness_evaluation(population_size,Num_requests,
                              Initial_population,Edge_servers,
                              Cloud_servers,Vehicle_position,
                              size_task,length_task,
                              vehicle_RSU_bw,RSU_cloud_bw,
                              Connected_RSU,edge_server_memory,
                              cloud_server_memory,edge_server_speed,
                              cloud_server_speed,edge_server_power,
                              cloud_server_power,swap_factor,
                              CPU_util_task,Emax)




while loop <= Generations:

    start_time_generation = datetime.datetime.now()

    selected_pop = selection(Num_requests,
                             population_size,
                             Initial_population,
                             fitness)
    

    pop_after_crss = crossover(population_size,selected_pop,cross_rate,
                               Num_requests,Edge_servers,Cloud_servers,
                               Vehicle_position, size_task,length_task,
                               Vehicle_RSU_bandwidth,RSU_cloud_bw,
                               Connected_RSU,edge_server_memory,cloud_server_memory,
                               edge_server_speed,cloud_server_speed,edge_server_power,
                               cloud_server_power,swap_factor,CPU_util_task,Emax)


    pop_aftermut = mutation(Num_requests, population_size, mutation_rate,
                        pop_after_crss,Connected_RSU,
                        Edge_servers,Cloud_servers)

    objective,fitness,energy,time,latency_violations,processing_violations,deadline_violations,Energy_request =  fitness_evaluation(population_size,Num_requests,
                                  pop_aftermut,Edge_servers,
                                  Cloud_servers,Vehicle_position,
                                  size_task,length_task,
                                  vehicle_RSU_bw,RSU_cloud_bw,
                                  Connected_RSU,edge_server_memory,
                                  cloud_server_memory,edge_server_speed,
                                  cloud_server_speed,edge_server_power,
                                  cloud_server_power,swap_factor,
                                  CPU_util_task,Emax)

    end_time_generation = datetime.datetime.now()
    time_per_generation = int((end_time_generation - start_time_generation).total_seconds() * 1000)

    best_solution_index = np.where(fitness == max(fitness))[0]
    if len(best_solution_index) > 1:
           best_solution_index = best_solution_index[:1]
    best_solution = pop_aftermut[best_solution_index]
    best_fitness_per_generation.append(fitness[best_solution_index].item())
    best_objective_per_generation.append(objective[best_solution_index].item())
    best_time_per_generation.append(time[best_solution_index].item())
    best_energy_per_generation.append(energy[best_solution_index].item())
    
    latency_per_generation.append(latency_violations[best_solution_index])
    processing_per_generation.append(processing_violations[best_solution_index])
    deadline_per_generation.append(deadline_violations[best_solution_index]) 

    total_latency_per_generation.append(np.sum(latency_violations[best_solution_index]))
    total_processing_per_generation.append(np.sum(processing_violations[best_solution_index]))
    total_deadline_per_generation.append(np.sum(deadline_violations[best_solution_index]))
    
    decision_time_per_generation.append(time_per_generation)

    Energy_consumption_per_request = Energy_request[best_solution_index]
    Initial_population = pop_aftermut

    loop = loop+1

    best_solution_index.tolist().clear()
    best_solution.tolist().clear()


end_time_genetic = datetime.datetime.now()
Total_time_genetic = int((end_time_genetic - start_time_genetic).total_seconds() * 1000)


normalized_fitness = (best_fitness_per_generation-np.min(best_fitness_per_generation))/(np.max(best_fitness_per_generation)-np.min(best_fitness_per_generation))

f1 = plt.figure(1)
plt.plot(normalized_fitness,'y*-')
plt.xlabel("Generation")
plt.ylabel("normalized_fitness")

f2 = plt.figure(2)
plt.plot(best_objective_per_generation,'r*-')
plt.xlabel("Generation")
plt.ylabel("Objective")

f3 = plt.figure(3)
plt.plot(best_time_per_generation,'bo-')
plt.xlabel("Generation")
plt.ylabel("Total execution time")

f4 = plt.figure(4)
plt.plot(best_energy_per_generation,'g+-')
plt.xlabel("Generation")
plt.ylabel("Total energy consumption")

f5 = plt.figure(5)
plt.plot(total_latency_per_generation,'m*-')
plt.xlabel("Generation")
plt.ylabel("Total latency violations")

f6 = plt.figure(6)
plt.plot(total_processing_per_generation,'ko-')
plt.xlabel("Generation")
plt.ylabel("Total processing violations")

f7 = plt.figure(7)
plt.plot(total_deadline_per_generation,'c+-')
plt.xlabel("Generation")
plt.ylabel("Total deadline violations")

#plt.show()

print(Energy_consumption_per_request)
#########################################################################################
#########################################################################################

