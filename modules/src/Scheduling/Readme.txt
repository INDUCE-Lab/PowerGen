
'''
Title:        PowerGen Toolkit
Description:  PowerGen (Resource Utilization and Power Generation Dataset Framework) Toolkit for Generating Resources Utlization and Corresponding Power Consumption in Edge and Cloud Computing Data Centers

Licence:      GPL - http://www.gnu.org/copyleft/gpl.html

Copyright (c) 2024, Intelligent Distributed Computing and Systems (INDUCE) Lab, The United Arab Emirates University, United Arab Emirates

If you are using any ideas, algorithms, packages, codes, datasets, workload, results, and plots, included in the scheduling directory please cite
the following paper:

https://doi.org/TBD">Leila Ismail, and Huned Materwala, "PowerGen: Resources Utilization
and Power Consumption Data Generation Framework for Energy Prediction in Edge and Cloud Computing", ANT 2024

'''

This directory contains the following files which are used to generate results for the scheduling scenario:

1) Energy_GA.py: This is the python code which uses genetic algorithm to schedule requests on edge or cloud servers in a way that the total energy consumption of edge and cloud servers is the minimum. This code uses a CPU-based linear power model to predict the power consumption of requests. This is an energy-aware approach. The code when executed prints the energy consumption of each request.

2) Random_Offloading.py - This is the python code that randomly schedules each request on either edge server (to which the request was submitted) or one of the cloud servers. This is a non-energy-aware approach. The code when executed prints the energy consumption of each request.

3) Energy consumption_10 requests.py - This is the python code that is used to plot the energy consumption per request using the energy-aware genetic algorithm approach and the random offloading approach. This code when executed will generate the 'Energy consumption_10 requests.svg' file.

4) Total energy consumption_increasing requests.py - This is the python code that is used to plot the total energy consumption with increasing requests using the energy-aware genetic algorithm approach and the random offloading approach. This code when executed will generate the 'Total energy consumption_increasing requests.svg' file.

5) Energy consumption.xlsx - This excel file contains the energy consumption of each request using energy-aware genetic algorithm and random offloading approaches. This excel file has 10 sheets. The first sheet contains the result for 10 requests, the second sheet contains the result for 20 requests, and so on. The number of requests is used as sheet name.

6) Energy consumption_10 requests.svg - This plot represents the energy consumption per request using energy-aware genetic algorithm and random offloading approaches. This plot is saved when the 'Energy consumption_10 requests.py' python code is executed.

7) Total energy consumption_increasing requests.svg - This plot represents the total energy consumption with increasing requests using energy-aware genetic algorithm and random offloading approaches. This plot is saved when the 'Total energy consumption_increasing requests.py' python code is executed.


----------------------------------------------
Steps to reproduce the results for scheduling scenarios
----------------------------------------------
1. Create a new excel file 'Energy consumption.xlsx'. Create 10 sheets in the file and name the sheets as 10, 20, ..., 100. In each sheet, create three columns with the following names: 'Request', 'Energy_GA', and 'Energy_random'. For the sheet having name '10' write numbers 1 to 10 in the 'Request' column representing the request index. For the sheet with name '20' write numbers 1 to 20 in the 'Request' column representing the request index. Similarly, do for the remaining sheets.

2. Open the 'Energy_GA.py' python file and set the value for 'Vehicles' to 10 on line 23. Execute the code. Copy the energy consumption for each request from the output of the code and paste them under the 'Energy_GA' column of the sheet '10' in the 'Energy consumption.xlsx' file.

3. Open the 'Random_Offloading.py' python file and set the value for 'Vehicles' to 10 on line 22. Execute the code. Copy the energy consumption for each request from the output of the code and paste them under the 'Energy_random' column of the sheet '10' in the 'Energy consumption.xlsx' file.

4. Repeat step 2 while changing the values for 'Vehicles' to 20, 30, 40, ..., 100 and pasting the results under 'Energy_GA' column of the sheet '20', '30', '40', ..., '100' respectively in the 'Energy consumption.xlsx' file.

5. Repeat step 3 while changing the values for 'Vehicles' to 20, 30, 40, ..., 100 and pasting the results under 'Energy_random' column of the sheet '20', '30', '40', ..., '100' respectively in the 'Energy consumption.xlsx' file.

6. For each sheet in the excel file compute the the total energy consumption for the 'Energy_GA' column by summing the energy consumption per request. 

7. For each sheet in the excel file compute the the total energy consumption for the 'Energy_random' column by summing the energy consumption per request.

8. Open the 'Energy consumption_10 requests.py' and copy the energy consumption per request for from the 'Energy_GA' column of the sheet '10' in the 'Energy_GA' list at line 8 of the python code.

9. Open the 'Energy consumption_10 requests.py' and copy the energy consumption per request for from the 'Energy_random' column of the sheet '10' in the 'Energy_random' list at line 11 of the python code.

10. Run the 'Energy consumption_10 requests.py' code.

11. Open the 'Total energy consumption_increasing requests.py' and copy the total energy consumption from the 'Energy_GA' column of the each sheet in the 'Energy_GA' list at line 8 of the python code.

12. Open the 'Total energy consumption_increasing requests.py' and copy the total energy consumption from the 'Energy_random' column of the each sheet in the 'Energy_random' list at line 11 of the python code.

13. Run the 'Total energy consumption_increasing requests.py' code.