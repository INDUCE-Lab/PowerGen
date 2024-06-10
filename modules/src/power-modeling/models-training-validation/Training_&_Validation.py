
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


'''This code develops the 8 power models used in the article.
It uses the training dataset to develop the models and the
validation dataset to validate the models. The models are evaluated
in terms of Mean Absolute Error (MAE). MAE values for the models are
written in an excel file. The actual and predicted power consumption
(using the 8 models) for the validation dataset are presented in a plot.
'''


'''Importing the libraries'''
import os
import pandas as pd
import numpy as np
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import joblib
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")


'''Set the path for dataset folder and folders to save the results'''
script_path = os.path.abspath(__file__) # Get the absolute path of the current script
code_folder = os.path.dirname(script_path) # Get the directory containing the script
data_folder = os.path.join(code_folder,'..','Datasets') # Construct the path to the dataset folder
plot_folder = os.path.join(code_folder,'..','Results_Plots') # Construct the path to the folder used to save the plot
metrics_folder = os.path.join(code_folder,'..','Results_Metrics') # Construct the path to the folder used to save the excel sheet containing MAE values
models_folder = os.path.join(code_folder,'..','Results_Developed models') # Construct the path to the folder used to save the developed models

data_file_path = os.path.join(data_folder, 'Training and validation dataset.csv') # Access dataset file in the Datasets folder
plot_file_path = os.path.join(plot_folder, 'Evaluation validation dataset.svg') # Name the plot file
metrics_file_path = os.path.join(metrics_folder, 'Metrics validation dataset.xlsx') # Name the excel sheet file


''' Set seed for reproducibility '''
seed = 42
np.random.seed(seed)
random_state = check_random_state(seed)


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
Pmin = min(df['POWER (W)']) # Define the minimum power consumption
Pmax = max(df['POWER (W)']) # Define the maximum power consumption
X_train_statistical_1, X_test_statistical_1, Y_train_statistical_1, Y_test_statistical_1 = train_test_split(X_statistical_1,Y_statistical_1, # Split the dataset for training (70%) and validation (30%)
                                                                                                             test_size=0.3,
                                                                                                             random_state=8)
def power_model_1(r, u, Pmin, Pmax):  # Define a function to compute power consumption
    return Pmin + ((Pmax - Pmin) * ((2 * u) - (u ** r)))
def objective_function(r, u, y, Pmin, Pmax):  # Define objective function of minimizing MAE to obtain optimal 'r' value
    predictions = power_model_1(r, u, Pmin, Pmax)
    return mean_absolute_error(y, predictions)
initial_value = 0 # Initial value for 'r
result = minimize(objective_function,initial_value, # Using minimize to find optimal 'r'
                  args=(X_train_statistical_1, Y_train_statistical_1, Pmin, Pmax))
optimal_params = result.x # Get the optimal alpha and beta values
r_optimal = optimal_params
joblib.dump((Pmin, Pmax, r_optimal), os.path.join(models_folder,  'Statistical Linear Regression_1.joblib')) # Save Statistical Linear Regression_1 model
prediction_statistical_1 = Pmin + ((Pmax - Pmin) * ((2 * X_test_statistical_1) - (X_test_statistical_1 ** r_optimal))) # Predict power consumption for validation dataset using the developed model
results['MAE'].append(metrics.mean_absolute_error(Y_test_statistical_1,prediction_statistical_1)) # Writing the MAE to the results dictionary


'''#######################################################################################
##########################################################################################
##########################################################################################
#######################################################################################'''

'''Model 2: Simple Linear Regression'''
X_SLR = df['CPU(%)'] # Define CPU as the independent variable
X_SLR = X_SLR.to_numpy()[:, None]
Y_SLR = df['POWER (W)'] # Define power consumption as the dependent variable
X_train_SLR, X_test_SLR, Y_train_SLR, Y_test_SLR = train_test_split(X_SLR,Y_SLR, # Split the dataset for training (70%) and validation (30%)
                                                    test_size=0.3,
                                                    random_state=8)
model_SLR = LinearRegression() # Define simple linear regression model
model_SLR.fit(X_train_SLR,Y_train_SLR) # Fit the model on the training dataset
joblib.dump(model_SLR, os.path.join(models_folder, 'Simple Linear Regression.joblib')) # Save the simple regression model
prediction_SLR = model_SLR.predict(X_test_SLR) # Predict power consumption for validation dataset using the developed model
results['MAE'].append(metrics.mean_absolute_error(Y_test_SLR,prediction_SLR)) # Writing the MAE to the results dictionary


'''#######################################################################################
##########################################################################################
##########################################################################################
#######################################################################################'''

'''Model 3: Polynomial Regression'''
X_PR_Deg3 = df['CPU(%)'] # Define CPU as the independent variable
X_PR_Deg3 = X_PR_Deg3.to_numpy()[:, None]
Y_PR_Deg3 = df['POWER (W)'] # Define power consumption as the dependent variable
PR_Deg3 = PolynomialFeatures(degree = 3) # Define polynomial regression model of 3rd degree
X_PR_Deg3 = PR_Deg3.fit_transform(X_PR_Deg3)
X_train_PR_Deg3, X_test_PR_Deg3, Y_train_PR_Deg3, Y_test_PR_Deg3 = train_test_split(X_PR_Deg3,Y_PR_Deg3, # Split the dataset for training (70%) and validation (30%)
                                                    test_size=0.3,
                                                    random_state=8)
model_PR_Deg3 = LinearRegression()
model_PR_Deg3.fit(X_train_PR_Deg3, Y_train_PR_Deg3) # Fit the model on the training dataset
joblib.dump(model_PR_Deg3, os.path.join(models_folder, 'Polynomial Regression.joblib')) # Save Polynomial Regression Degree 3 model
prediction_PR_Deg3 = model_PR_Deg3.predict(X_test_PR_Deg3) # Predict power consumption for validation dataset using the developed model
results['MAE'].append(metrics.mean_absolute_error(Y_test_PR_Deg3,prediction_PR_Deg3)) # Writing the MAE to the results dictionary


'''#######################################################################################
##########################################################################################
##########################################################################################
#######################################################################################'''

'''Model 4: Statistical Linear Regression_2'''
X_statistical_2 = df['CPU(%)'] # Define CPU as the independent variable
X_statistical_2 = X_statistical_2.to_numpy()[:, None]
Y_statistical_2 = df['POWER (W)'] # Define power consumption as the dependent variable
Pmin = min(df['POWER (W)']) # Define the minimum power consumption
Pmax = max(df['POWER (W)']) # Define the maximum power consumption
X_train_statistical_2, X_test_statistical_2, Y_train_statistical_2, Y_test_statistical_2 = train_test_split(X_statistical_2,Y_statistical_2, # Split the dataset for training (70%) and validation (30%)
                                                                                                             test_size=0.3,
                                                                                                             random_state=8)
def power_model_4(params, u): # Define a function to compute power consumption
    alpha, beta = params
    return Pmin + (Pmax - Pmin) * alpha * (u**beta)
def objective_function(params): # Define objective function of minimizing MAE to obtain optimal alpha and beta values
    alpha, beta = params
    predictions = power_model_4((alpha, beta), X_train_statistical_2)
    return mean_absolute_error(Y_train_statistical_2, predictions)
initial_values = [0.01, 0.01] # Initial values for alpha and beta
result = minimize(objective_function, initial_values) # Minimize the objective function to find optimal alpha and beta values
optimal_params = result.x # Get the optimal alpha and beta values
alpha_optimal, beta_optimal = optimal_params
joblib.dump((Pmin, Pmax, alpha_optimal, beta_optimal), os.path.join(models_folder,  'Statistical Linear Regression_2.joblib')) # Save Statistical Linear Regression_1 model
prediction_statistical_2 = Pmin + (Pmax - Pmin) * alpha_optimal * (X_test_statistical_2 ** beta_optimal) # Predict power consumption for validation dataset using the developed model
results['MAE'].append(metrics.mean_absolute_error(Y_test_statistical_2,prediction_statistical_2)) # Writing the MAE to the results dictionary


'''#######################################################################################
##########################################################################################
##########################################################################################
#######################################################################################'''

'''Model 5: Support Vector Regression'''
X_SVR = df[['CPU(%)', 'MEMORY(%)']] # Define CPU and memory as the independent variables
Y_SVR = df['POWER (W)'] # Define power consumption as the dependent variable
X_train_SVR, X_test_SVR, Y_train_SVR, Y_test_SVR = train_test_split(X_SVR,Y_SVR, # Split the dataset for training (70%) and validation (30%)
                                                    test_size=0.3,
                                                    random_state=8)
model_SVR = SVR(kernel = 'poly',degree=2,gamma=0.01,coef0=100,C=100) # Define support vector regression model
model_SVR.fit(X_train_SVR, Y_train_SVR) # Fit the model on the training dataset
joblib.dump(model_SVR, os.path.join(models_folder, 'Support Vector Regression.joblib')) # Save Support Vector Regression model
prediction_SVR = model_SVR.predict(X_test_SVR) # Predict power consumption for validation dataset using the developed model
results['MAE'].append(metrics.mean_absolute_error(Y_test_SVR,prediction_SVR)) # Writing the MAE to the results dictionary


'''#######################################################################################
##########################################################################################
##########################################################################################
#######################################################################################'''

'''Model 6: Multi Linear Regression (3 features)'''
X_MLR_3 = df[['CPU(%)', 'MEMORY(%)', 'DISK (r-wr/s)']] # Define CPU, memory, and disk as the independent variables
Y_MLR_3 = df['POWER (W)'] # Define power consumption as the dependent variable
X_train_MLR_3, X_test_MLR_3, Y_train_MLR_3, Y_test_MLR_3 = train_test_split(X_MLR_3,Y_MLR_3, # Split the dataset for training (70%) and validation (30%)
                                                    test_size=0.3,
                                                    random_state=8)
model_MLR_3 = LinearRegression() # Define multi linear regression model
model_MLR_3.fit(X_train_MLR_3,Y_train_MLR_3) # Fit the model on the training dataset
joblib.dump(model_MLR_3, os.path.join(models_folder, 'Multi Linear Regression (3 features).joblib')) # Save Multi Linear Regression (3 features) model
prediction_MLR_3 = model_MLR_3.predict(X_test_MLR_3) # Predict power consumption for validation dataset using the developed model
results['MAE'].append(metrics.mean_absolute_error(Y_test_MLR_3,prediction_MLR_3)) # Writing the MAE to the results dictionary


'''#######################################################################################
##########################################################################################
##########################################################################################
#######################################################################################'''

'''Model 7: Multi Linear Regression (4 features)'''
X_MLR_4 = df[['CPU(%)', 'MEMORY(%)', 'DISK (r-wr/s)', 'NETWORK (bits/s)']] # Define CPU, memory, disk, and network as the independent variables
Y_MLR_4 = df['POWER (W)'] # Define power consumption as the dependent variable
X_train_MLR_4, X_test_MLR_4, Y_train_MLR_4, Y_test_MLR_4 = train_test_split(X_MLR_4,Y_MLR_4, # Split the dataset for training (70%) and validation (30%)
                                                    test_size=0.3,
                                                    random_state=8)
model_MLR_4 = LinearRegression() # Define multi linear regression model
model_MLR_4.fit(X_train_MLR_4,Y_train_MLR_4) # Fit the model on the training dataset
joblib.dump(model_MLR_4, os.path.join(models_folder, 'Multi Linear Regression (4 features).joblib')) # Save Multi Linear Regression (4 features) model
prediction_MLR_4 = model_MLR_4.predict(X_test_MLR_4) # Predict power consumption for validation dataset using the developed model
results['MAE'].append(metrics.mean_absolute_error(Y_test_MLR_4,prediction_MLR_4)) # Writing the MAE to the results dictionary
Intercept_MLR_4 = model_MLR_4.intercept_ # Save the intercept for the developed multi linear regression (4 features) model
Slope_MLR_4 = model_MLR_4.coef_ # Save the slope for the developed multi linear regression (4 features) model


'''#######################################################################################
##########################################################################################
##########################################################################################
#######################################################################################'''

'''Model 8: Multi Linear Regression with Fixed Intercept (4 features)'''
new_Intercept_MLR = min(df['POWER (W)']) # Define miminum power consumption as the intercept
model_MLR_4.intercept_ = new_Intercept_MLR # Replace the intercept of MLR model with the minimum power consumption
joblib.dump(model_MLR_4, os.path.join(models_folder, 'Multi Linear Regression with Fixed Intercept (4 features).joblib')) # Save Multi Linear Regression with Fixed Intercept (4 features) model
prediction_MLR_4_Fixed_Intercept = model_MLR_4.predict(X_test_MLR_4) # Predict power consumption for validation dataset using the developed model
results['MAE'].append(metrics.mean_absolute_error(Y_test_MLR_4,prediction_MLR_4_Fixed_Intercept)) # Writing the MAE to the results dictionary


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
Sorting the dataframe based on actual power consumption in an ascending order '''
df_results = pd.DataFrame()
df_results['Actual'] = Y_test_SLR.tolist()
df_results['Model 1'] = list(np.concatenate(prediction_statistical_1.tolist()))
df_results['Model 2'] = prediction_SLR
df_results['Model 3'] = prediction_PR_Deg3.tolist()
df_results['Model 4'] = list(np.concatenate(prediction_statistical_2.tolist()))
df_results['Model 5'] = prediction_SVR.tolist()
df_results['Model 6'] = prediction_MLR_3.tolist()
df_results['Model 7'] = prediction_MLR_4.tolist()
df_results['Model 8'] = prediction_MLR_4_Fixed_Intercept.tolist()

df_results = df_results.sort_values(by="Actual",ascending=True,ignore_index=True)

####### We will plot the actual values and predicted values using all models.  This plot will be then saved
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,6))

plt.plot(df_results['Actual'], color='blue', linestyle='-', marker = 'o', label = 'Actual power', linewidth=2, markersize=10)
plt.plot(df_results['Model 1'], color='orange', linestyle='-', marker = 'v', label = 'Predicted power - Model 1', linewidth=2, markersize=10)
plt.plot(df_results['Model 2'], color='green', linestyle='-', marker = '^', label = 'Predicted power - Model 2', linewidth=2, markersize=10)
plt.plot(df_results['Model 3'], color='red', linestyle='-', marker = '1', label = 'Predicted power - Model 3', linewidth=2, markersize=10)
plt.plot(df_results['Model 4'], color='purple', linestyle='-', marker = '2', label = 'Predicted power - Model 4', linewidth=2, markersize=10)
plt.plot(df_results['Model 5'], color='brown', linestyle='-', marker = '<', label = 'Predicted power - Model 5', linewidth=2, markersize=10)
plt.plot(df_results['Model 6'], color='gray', linestyle='-', marker = '>', label = 'Predicted power - Model 6', linewidth=2, markersize=10)
plt.plot(df_results['Model 7'], color='olive', linestyle='-', marker = '*', label = 'Predicted power - Model 7', linewidth=2, markersize=10)
plt.plot(df_results['Model 8'], color='cyan', linestyle='-', marker = 's', label = 'Predicted power - Model 8', linewidth=2, markersize=10)


plt.ylabel('Power consumption (Watts)', fontsize=18)
plt.xlabel('Validation dataset points', fontsize=18)
ax=plt.gca()
ax.tick_params(axis='both', which='major', labelsize=16)
plt.legend(fontsize=15, loc='best')
plt.savefig(plot_file_path, format='svg', dpi=800, bbox_inches='tight')
plt.show()

