
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


Requests = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

Energy_GA = [1065.899948, 2930.510779, 6919.528199, 9257.042761, 14405.13373,
             20449.9174, 29659.71964, 36076.22125, 50036.52536, 57551.9361]

Energy_random = [3404.501961, 6887.197205, 13472.37998, 35710.70645, 32896.0845,
                 43281.21703, 79615.89013, 78584.40418, 102326.3369, 120976.6363]





####### We will plot the actual values and predicted values using all models.  This plot will be then saved
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,6))

plt.plot(Requests, Energy_GA, color='b', linestyle='--', marker = 'o', label = 'Energy-aware scheduling', linewidth=2, markersize=10)
plt.plot(Requests, Energy_random, color='r', linestyle='-.', marker = '*', label = 'Random scheduling', linewidth=2, markersize=10)


plt.ylabel('Total energy consumption (Watts-seconds)', fontsize=18)
plt.xlabel('Requests', fontsize=18)
ax=plt.gca()
plt.xticks(range(min(Requests), max(Requests)+1, 10))
ax.tick_params(axis='both', which='major', labelsize=18)
plt.grid()
plt.legend(fontsize=18, loc='best')
plt.savefig('Total energy consumption_increasing requests.svg', format='svg', dpi=800, bbox_inches='tight')
plt.show()

