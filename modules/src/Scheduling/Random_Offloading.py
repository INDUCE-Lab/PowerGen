
'''
Title:        PowerGen Toolkit

Description:  PowerGen (Power Generation Dataset) Toolkit for Generating Resources Utlization and Corresponding
Power Consumption in Edge and Cloud Computing Data Centers

Licence:      GPL - http://www.gnu.org/copyleft/gpl.html

Copyright (c) 2024, Intelligent Distributed Computing and Systems (INDUCE) Lab, The United Arab Emirates University, United Arab Emirates

If you are using any ideas, algorithms, packages, codes, datasets, workload, results, and plots, included in the scheduling directory please cite
the following paper:

https://doi.org/TBD">Leila Ismail, and Huned Materwala, "PowerGen: Resources Utilization
and Power Consumption Data Generation Framework for Energy Prediction in Edge and Cloud Computing",
ANT 2024

'''

'''
This is the python code that randomly schedules each request on either edge server (to which the request was submitted)
or one of the cloud servers. This is a non-energy-aware approach. The code when executed prints the energy consumption
of each request.
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
deadline = latency+processing
latency_req = [latency] * Num_requests #Array to store latency requirements of the requests
processing_req = [processing] * Num_requests #Array to store latency requirements of the requests
deadline_req = [deadline] * Num_requests #Array to store latency requirements of the requests



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
CPU_util_task = np.random.uniform(Minimum_CPU,Maximum_CPU,Num_requests) #generating CPU utilization for 'Num_tasks' tasks uniformly between min_CPU and max_CPU

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
bandwidth between vehicle - Roadside Units (edge servers) and between Roadside Units (edge servers) - the cloud servers. A constant bandwidth is considered throughout the simulation
in this version of the toolkit.  Bandwidth between each vehicle and Roadside Unit (edge server) is the same, and the bandwidth between each Roadside Unit
(edge server) and the cloud are different.
"""

vehicle_RSU_bw = Vehicle_RSU_bandwidth #generating a constant bandwidth of 'Vehicle_RSU_bandwidth' Megabits/second between each vehicle and connected Roadside Unit
RSU_cloud_bw = np.random.randint(Minimum_edgeCloudBW, Maximum_edgeCloudBW, size=(Edge_servers))    #generating random Roadside unit - cloud bandwidth between Minimum_edgeCloudBW and Maximum_edgeCloudBW Megabits for each RSU and cloud server


#########################################################################################
#########################################################################################

"""
This function simulates the RoadSide Units.  Each vehicle communicates to a RoadSide Unit and each Roadside Unit is equipped with an edge server.
Consequently, the number of RoadSide Units are and the edge servers are equal in the simulated network. The edge servers are in the same location
as their corresponding RoadSide Units.  RoadSide Units are placed equidistant.
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

server_alloc = [0] * Num_requests #initializing an array to store server allocation of the requests
comm_time = [0] * Num_requests #initializing an array to store communication time of the requests
wait_time = [0] * Num_requests #initializing an array to store waiting time of the requests
proc_time = [0] * Num_requests #initializing an array to store processing time of the requests
io_time = [0] * Num_requests #initializing an array to store i/o time of the requests
exec_time = [0] * Num_requests #initializing an array to store execution time of the requests
decision_time = [0] * Num_requests #initializing an array to store decision making time of the requests
power_consp = [0] * Num_requests #initializing an array to store power consumption of the requests
energy_consp = [0] * Num_requests #initializing an array to store energy consumption of the requests


start = datetime.datetime.now()

Allocated_random_server = [0] * Num_requests #initializing an array to store random server selection for each request
available_cloud_servers = list(range(Edge_servers, Edge_servers+Cloud_servers)) # initalizing a list of all available cloud servers for scheduling

for request in range(0, Num_requests):
    available_edge_server = Connected_RSU[request] # defining the available edge server for each request
    available_cloud_servers.append(available_edge_server)  # defining list of available edge and cloud servers for each request
    Allocated_random_server[request] = random.choice(available_cloud_servers) # randomly allocating a server to each request


Tasks_in_server = {}
for server in range(0, Edge_servers+Cloud_servers):
    Tasks_in_server[server] = []

for request in range(0, Num_requests):
    Tasks_in_server[Allocated_random_server[request]].append(request)

for server in range(0, Edge_servers+Cloud_servers):
    if len(Tasks_in_server[server]) != 0:

        server_alloc[request] = server+1

        if server < Edge_servers: # if it is an edge server
        
            sum_wait = 0
            wait_time_temp = 0
            for request in Tasks_in_server[server]:

                if Vehicle_position[request] == 0: # if vehicle is in the range
                    comm_time[request] = (size_task[request]/vehicle_RSU_bw +
                                          (0.01*size_task[request])/vehicle_RSU_bw)

                elif Vehicle_position[request] == 1: # if vehicle is not in the range
                    comm_time[request] = (size_task[request]/vehicle_RSU_bw + (0.01*size_task[request])/RSU_cloud_bw[Connected_RSU[request]] +
                                          (0.01*size_task[request])/RSU_cloud_bw[random.choice(list(range(0,Edge_servers)))] +
                                          (0.01*size_task[request])/vehicle_RSU_bw)

                wait_time[request] = wait_time_temp
                running_time = length_task[request]/edge_server_speed[server]
                proc_time_temp = running_time + wait_time_temp
                wait_time_temp = proc_time_temp
                sum_wait = wait_time_temp + sum_wait
                proc_time[request] = proc_time_temp

                if math.ceil(size_task[request]/edge_server_memory[server]) == 0:
                    io_time[request] = 0
                else:
                    io_time[request] = swap_factor*((math.ceil(size_task[request]/edge_server_memory[server]))-1)

                power_consp[request] = (edge_server_power[server][low_CPU[request]] +
                                        ((edge_server_power[server][low_CPU[request]] - edge_server_power[server][up_CPU[request]])*
                                         (CPU_util_task[request] - low_CPU[request]*10)/10))
                
                energy_consp[request] = power_consp[request]*(proc_time[request]+io_time[request])

                exec_time[request] = (comm_time[request]+
                                      proc_time[request]+
                                      io_time[request])

        else:

            sum_wait = 0
            wait_time_temp = 0
            for request in Tasks_in_server[server]:

                if Vehicle_position[request] == 0: # if vehicle is in the range
                    comm_time[request] = (size_task[request]/vehicle_RSU_bw +
                                          size_task[request]/RSU_cloud_bw[Connected_RSU[request]] +
                                          (0.01*size_task[request])/RSU_cloud_bw[Connected_RSU[request]] +
                                          (0.01*size_task[request])/vehicle_RSU_bw)

                elif Vehicle_position[request] == 1: # if vehicle is not in the range
                    comm_time[request] = (size_task[request]/vehicle_RSU_bw +
                                          size_task[request]/RSU_cloud_bw[Connected_RSU[request]] +
                                          (0.01*size_task[request])/RSU_cloud_bw[random.choice(list(range(0,Edge_servers)))] +
                                          (0.01*size_task[request])/vehicle_RSU_bw)

                wait_time[request] = wait_time_temp
                running_time = length_task[request]/cloud_server_speed[server-Edge_servers]
                proc_time_temp = running_time + wait_time_temp
                wait_time_temp = proc_time_temp
                sum_wait = wait_time_temp + sum_wait
                proc_time[request] = proc_time_temp

                if math.ceil(size_task[request]/cloud_server_memory[server-Edge_servers]) == 0:
                    io_time[request] = 0
                else:
                    io_time[request] = swap_factor*((math.ceil(size_task[request]/cloud_server_memory[server-Edge_servers]))-1)

                power_consp[request] = (cloud_server_power[server-Edge_servers][low_CPU[request]] +
                                        ((cloud_server_power[server-Edge_servers][low_CPU[request]] - cloud_server_power[server-Edge_servers][up_CPU[request]])*
                                         (CPU_util_task[request] - low_CPU[request]*10)/10))
                
                energy_consp[request] = power_consp[request]*(proc_time[request]+io_time[request])

                exec_time[request] = (comm_time[request]+
                                      proc_time[request]+
                                      io_time[request])
            

end = datetime.datetime.now()

delta = end-start
time_to_decision = int(delta.total_seconds() * 1000) # millisecond
decision_time = [time_to_decision] * Num_requests


Tasks = list(range(1,Num_requests+1))
latency_violations = []
processing_violations = []
deadline_violations = []

for i in range(0, Num_requests):
    if comm_time[i] > latency:
        latency_violations.append(comm_time[i] - latency_req[i])
    else:
        latency_violations.append(0)
        
    if proc_time[i] > processing:
        processing_violations.append(proc_time[i] - processing_req[i])
    else:
        processing_violations.append(0)
        
    if exec_time[i] > deadline:
        deadline_violations.append(exec_time[i] - deadline_req[i])
    else:
        deadline_violations.append(0)

##with open("Random_offloading_req(50).csv", "wt", newline='') as file:
##    writer = csv.writer(file)
##    writer.writerow(["Request","Server",
##                     "Power", "Energy",
##                     "Communication time","Processing time","i/o time","Execution time",
##                     "Latency requirement","Processing requirement","Deadline",
##                     "Latency violations","Processing violations","Deadline violations",
##                     "Decision making time"])
##    for row in zip(Tasks,server_alloc,
##                   power_consp,energy_consp,
##                   comm_time,proc_time,io_time,exec_time,
##                   latency_req,processing_req,deadline_req,
##                   latency_violations,processing_violations,deadline_violations,
##                   decision_time):
##        writer.writerow(row)
