
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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn import metrics
import joblib
import os
import warnings
warnings.filterwarnings("ignore")


script_path = os.path.abspath(__file__) # Get the absolute path of the current script
code_folder = os.path.dirname(script_path) # Get the directory containing the script (code folder in this case)
data_folder = os.path.join(code_folder,'..','Datasets') # Construct the path to the data folder
plot_folder = os.path.join(code_folder,'..','Results_Plots') # Construct the path to the plot folder
metrics_folder = os.path.join(code_folder,'..','Results_Metrics') # Construct the path to the metrics folder
models_folder = os.path.join(code_folder,'..','Results_Developed models') # Construct the path to the developed models folder

data_file_path = os.path.join(data_folder, 'Testing dataset_Sysbench.csv') # Access dataset file in the Datasets folder without hardcoding the path
plot_file_path = os.path.join(plot_folder, 'Evaluation testing dataset_Sysbench.svg')
metrics_file_path = os.path.join(metrics_folder, 'Metrics testing dataset_Sysbench.xlsx')


'''Create a dictionary to store the results'''
results = {
    'Models': ['Statistical Linear Regression_1',
               'Simple Linear Regression',
               'Polynomial Regression',
               'Statistical Linear Regression_2',
               'Support Vector Regression',
               'Multi Linear Regression (3 features)',
               'Multi Linear Regression (4 features)',
               'Multi Linear Regression with Fixed Intercept (4 features)'],
    'MAE': [],
}

'''Import the dataset'''
df = pd.read_csv(data_file_path)


'''#######################################################################################
##########################################################################################
##########################################################################################
#######################################################################################'''

'''Model 1: Statistical Linear Regression_1'''
X_statistical_1 = df['CPU(%)'] # Define CPU as the independent variable
X_statistical_1 = X_statistical_1.to_numpy()[:, None]
Y_statistical_1 = df['POWER (W)'] # Define power consumption as the dependent variable
statistical_1_model = joblib.load(os.path.join(models_folder, 'Statistical Linear Regression_1.joblib'))
def power_model_1(r, u, Pmin, Pmax):  # Define a function to compute power consumption
    return Pmin + ((Pmax - Pmin) * ((2 * u) - (u ** r)))
prediction_statistical_1 = power_model_1(statistical_1_model[2],
                                         X_statistical_1,
                                         statistical_1_model[0],
                                         statistical_1_model[1])
results['MAE'].append(metrics.mean_absolute_error(Y_statistical_1,prediction_statistical_1)) # Writing the MAE to the results dictionary


'''#######################################################################################
##########################################################################################
##########################################################################################
#######################################################################################'''

'''Model 2: Simple Linear Regression'''
x_test_SLR = df['CPU(%)']
x_test_SLR = x_test_SLR.to_numpy()[:, None]
y_test_SLR = df['POWER (W)']
SLR_model = joblib.load(os.path.join(models_folder, 'Simple Linear Regression.joblib'))
prediction_SLR = SLR_model.predict(x_test_SLR)
results['MAE'].append(metrics.mean_absolute_error(y_test_SLR,prediction_SLR)) # Writing the MAE to the results dictionary


'''#######################################################################################
##########################################################################################
##########################################################################################
#######################################################################################'''

'''Model 3: Polynomial Regression'''
x_test_PR_Deg3 = df['CPU(%)']
x_test_PR_Deg3 = x_test_PR_Deg3.to_numpy()[:, None]
y_test_PR_Deg3 = df['POWER (W)']
PR_Deg3 = PolynomialFeatures(degree = 3)
x_test_PR_Deg3 = PR_Deg3.fit_transform(x_test_PR_Deg3)
PR3_model = joblib.load(os.path.join(models_folder, 'Polynomial Regression.joblib'))
prediction_PR_Deg3 = PR3_model.predict(x_test_PR_Deg3)
results['MAE'].append(metrics.mean_absolute_error(y_test_PR_Deg3,prediction_PR_Deg3)) # Writing the MAE to the results dictionary


'''#######################################################################################
##########################################################################################
##########################################################################################
#######################################################################################'''

'''Model 4: Statistical Linear Regression_2'''
X_statistical_2 = df['CPU(%)'] # Define CPU as the independent variable
X_statistical_2 = X_statistical_2.to_numpy()[:, None]
Y_statistical_2 = df['POWER (W)'] # Define power consumption as the dependent variable
statistical_2_model = joblib.load(os.path.join(models_folder, 'Statistical Linear Regression_2.joblib'))
def power_model_4(params, u): # Define a function to compute power consumption
    alpha, beta = params
    return Pmin + (Pmax - Pmin) * alpha * (u**beta)
Pmin=130.4966
Pmax=209.644
prediction_statistical_2 = power_model_4((statistical_2_model[2],
                                          statistical_2_model[3]),
                                         X_statistical_2)
results['MAE'].append(metrics.mean_absolute_error(Y_statistical_2,prediction_statistical_2)) # Writing the MAE to the results dictionary


'''#######################################################################################
##########################################################################################
##########################################################################################
#######################################################################################'''

'''Model 5: Support Vector Regression'''
x_test_SVR = df[['CPU(%)', 'MEMORY(%)']]
y_test_SVR = df['POWER (W)']
SVR_model = joblib.load(os.path.join(models_folder, 'Support Vector Regression.joblib'))
prediction_SVR = SVR_model.predict(x_test_SVR)
results['MAE'].append(metrics.mean_absolute_error(y_test_SVR,prediction_SVR)) # Writing the MAE to the results dictionary


'''#######################################################################################
##########################################################################################
##########################################################################################
#######################################################################################'''

'''Model 6: Multi Linear Regression (3 features)'''
x_test_MLR = df[['CPU(%)', 'MEMORY(%)', 'DISK (r-wr/s)']]
y_test_MLR = df['POWER (W)']
MLR_model = joblib.load(os.path.join(models_folder, 'Multi Linear Regression (3 features).joblib'))
prediction_MLR = MLR_model.predict(x_test_MLR)
results['MAE'].append(metrics.mean_absolute_error(y_test_MLR,prediction_MLR)) # Writing the MAE to the results dictionary


'''#######################################################################################
##########################################################################################
##########################################################################################
#######################################################################################'''

'''Model 7: Multi Linear Regression (4 features)'''
x_test_MLR_2 = df[['CPU(%)', 'MEMORY(%)', 'DISK (r-wr/s)', 'NETWORK (bits/s)']]
y_test_MLR_2 = df['POWER (W)']                                                    
MLR2_model = joblib.load(os.path.join(models_folder, 'Multi Linear Regression (4 features).joblib'))
prediction_MLR_2 = MLR2_model.predict(x_test_MLR_2)
results['MAE'].append(metrics.mean_absolute_error(y_test_MLR_2,prediction_MLR_2)) # Writing the MAE to the results dictionary


'''#######################################################################################
##########################################################################################
##########################################################################################
#######################################################################################'''

'''Model 8: Multi Linear Regression with Fixed Intercept (4 features)'''
MLR_model_fixed_intercept = joblib.load(os.path.join(models_folder, 'Multi Linear Regression with Fixed Intercept (4 features).joblib'))
prediction_MLR_Fixed_Intercept = MLR_model_fixed_intercept.predict(x_test_MLR_2)
results['MAE'].append(metrics.mean_absolute_error(y_test_MLR_2,prediction_MLR_Fixed_Intercept)) # Writing the MAE to the results dictionary


'''#######################################################################################
##########################################################################################
##########################################################################################
#######################################################################################'''

df_metrics = pd.DataFrame(results) # Create a dataframe from the results dictionary
df_metrics.to_excel(metrics_file_path, index=False) # Export the dataframe to Excel


'''#######################################################################################
##########################################################################################
##########################################################################################
#######################################################################################'''

''' Creating a dataframe of actual and predicted power values.
Sorting the dataframe based on actual power consumption in an ascending order'''
df_results = pd.DataFrame()
df_results = pd.DataFrame()
df_results['Actual'] = y_test_SLR.tolist()
df_results['Model 1'] = list(np.concatenate(prediction_statistical_1.tolist()))
df_results['Model 2'] = prediction_SLR
df_results['Model 3'] = prediction_PR_Deg3.tolist()
df_results['Model 4'] = list(np.concatenate(prediction_statistical_2.tolist()))
df_results['Model 5'] = prediction_SVR.tolist()
df_results['Model 6'] = prediction_MLR.tolist()
df_results['Model 7'] = prediction_MLR_2.tolist()
df_results['Model 8'] = prediction_MLR_Fixed_Intercept.tolist()

df_results = df_results.sort_values(by="Actual",ascending=True,ignore_index=True)

####### We will plot the actual values and predicted values using all models.  This plot will be then saveddf_results = pd.DataFrame()
Actual = y_test_SLR.tolist()
Model_1 = list(np.concatenate(prediction_statistical_1.tolist()))
Model_2 = prediction_SLR
Model_3 = prediction_PR_Deg3.tolist()
Model_4 = list(np.concatenate(prediction_statistical_2.tolist()))
Model_5 = prediction_SVR.tolist()
Model_6 = prediction_MLR.tolist()
Model_7 = prediction_MLR_2.tolist()
Model_8 = prediction_MLR_Fixed_Intercept.tolist()

models = ['Predicted power - Model 1', 'Predicted power - Model 2',
          'Predicted power - Model 3', 'Predicted power - Model 4', 'Predicted power - Model 5',
          'Predicted power - Model 6', 'Predicted power - Model 7', 'Predicted power - Model 8']

power = [Model_1, Model_2, Model_3,Model_4,
         Model_5,Model_6,Model_7,Model_8]

power = [list(x) for x in power]

power_final=[]
  
for l in power:
    power_final += l

plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(10,6))

plt.bar(models,power_final)
plt.axhline(y=Actual, color='red', linewidth=3, linestyle='dashed', label= 'Actual power')
plt.ylabel('Power consumption (Watts)', fontsize=18)
plt.xlabel('Models', fontsize=18)
ax=plt.gca()
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xticklabels(models, rotation=25, ha='right')
plt.legend(fontsize=15, loc='best')
plt.savefig(plot_file_path, format='svg', dpi=800, bbox_inches='tight')
plt.show()
