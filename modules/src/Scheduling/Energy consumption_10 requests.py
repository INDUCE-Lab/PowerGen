
'''
Title:        PowerGen Toolkit
Description:  PowerGen (Power Generation Dataset) Toolkit for Generating Resources Utlization and Corresponding Power Consumption in Edge and Cloud Computing Data Centers
Licence:      GPL - http://www.gnu.org/copyleft/gpl.html

Copyright (c) 2024, Intelligent Distributed Computing and Systems (INDUCE) Lab, The United Arab Emirates University, United Arab Emirates

If you are using any ideas, algorithms, packages, codes, datasets, workload, results, and plots, included in the power-modeling directory please cite
the following paper:

https://doi.org/TBD">Leila Ismail, and Huned Materwala, "PowerGen: Resources Utilization
and Power Consumption Data Generation Framework for Energy Prediction in Edge and Cloud Computing",
ANT 2024

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


Requests = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Energy_GA = [75.05852695, 54.53864533, 57.97488724, 80.50775951, 74.87312463,
             172.7374799, 352.9509963, 82.08166085, 21.80269334, 93.37417397]

Energy_random = [75.05852695, 150.2233757, 101.7919501, 314.3762475, 199.9192144,
                 386.1613241, 764.1104611, 972.605426, 251.243561, 189.0118737]





####### We will plot the actual values and predicted values using all models.  This plot will be then saved
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,6))

plt.plot(Requests, Energy_GA, color='b', linestyle='--', marker = 'o', label = 'Energy-aware scheduling', linewidth=2, markersize=10)
plt.plot(Requests, Energy_random, color='r', linestyle='-.', marker = '*', label = 'Random scheduling', linewidth=2, markersize=10)


plt.ylabel('Energy consumption (Watts-seconds)', fontsize=18)
plt.xlabel('Request index', fontsize=18)
ax=plt.gca()
ax.tick_params(axis='both', which='major', labelsize=18)
plt.legend(fontsize=18, loc='best')
plt.xticks(range(min(Requests), max(Requests)+1));
plt.grid()
plt.savefig('Energy consumption_10 requests.svg', format='svg', dpi=800, bbox_inches='tight')
plt.show()

