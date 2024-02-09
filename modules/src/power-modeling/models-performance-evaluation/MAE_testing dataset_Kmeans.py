
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

'''Importing the libraries'''
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import math
import os

'''Set the path for dataset folder and folders to save the results'''
script_path = os.path.abspath(__file__) # Get the absolute path of the current script
code_folder = os.path.dirname(script_path) # Get the directory containing the script (code folder in this case)
plot_folder = os.path.join(code_folder,'..','Results_Plots') # Construct the path to the plot folder
metrics_folder = os.path.join(code_folder,'..','Results_Metrics') # Construct the path to the metrics folder

plot_file_path = os.path.join(plot_folder, 'MAE testing dataset_Kmeans.svg')
metrics_file_path = os.path.join(metrics_folder, 'Metrics testing dataset_kmeans.xlsx')


'''Import the metrics file'''
df = pd.read_excel(metrics_file_path)


MAE = df['MAE'].tolist()
MAE= [float(i)/max(MAE) for i in MAE]

# repeat the first value to close the circular graph
MAE.append(MAE[0])


angles = [n / float(len(MAE)) * 2 * pi for n in range(len(MAE)-1)]
angles += angles[:1]

labels = ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5", "Model 6", "Model 7", "Model 8", "Model 1"]
          
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12),subplot_kw=dict(polar=True))
plt.rcParams.update({'font.size': 18})
plt.xticks(angles, labels, size=30)

maximum = max(MAE)
plt.yticks(np.arange(1, maximum+1))
ax.set_yticklabels([])

ax.set_rlabel_position(30)


# Go through labels and adjust alignment based on where it is in the circle.
for label, angle in zip(ax.get_xticklabels(), angles):
    if angle < 1.6755160819145563:
        label.set_horizontalalignment('left')
    elif angle ==1.6755160819145563:
        label.set_horizontalalignment('center')
        label.set_verticalalignment('bottom')
    elif 1.6755160819145563 < angle < 4.2:
        label.set_horizontalalignment('right')
    elif angle == 4.607669225265029:
        label.set_horizontalalignment('center')
        label.set_verticalalignment('top')
    else:
        label.set_horizontalalignment('left')
        

# part 1
ax.plot(angles, MAE, linewidth=2, color='blue', linestyle='solid')
ax.fill(angles, MAE, 'skyblue', alpha=0.4)
##for ti, di in zip(angles, MAE):
##        ax.text(ti, di+1, di, size=22, color='red', ha='right', va='top')
ax.scatter(angles, MAE, color='red', s=85)
plt.savefig(plot_file_path, format='svg', dpi=300, bbox_inches='tight')
plt.show()

