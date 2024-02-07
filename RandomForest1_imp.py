#This file contains the execution of random forest algorithm. It includes spltting data into train and test data too.
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import (precision_score, recall_score, f1_score, classification_report,
                             accuracy_score, mean_squared_error, mean_absolute_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

m2 = pd.read_csv('m2_1.csv', low_memory=False)

num_features = 21

#Constant Y value (target variable)
constant_y = m2['rating_pinfo']

#Separate features (X) and target variable (y)
X = m2.drop('rating_pinfo', axis=1)
y = m2['rating_pinfo']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
#X_test.to_csv('X_test.csv', index=False)

#Create a Random Forest Regressor
#rf_regressor = RandomForestRegressor(n_estimators=100, random_state=50)
rf_regressor = RandomForestRegressor(n_estimators=200, max_depth = 16, random_state=50)
print("Step1 complete")
#Train the model
rf_regressor.fit(X_train, y_train)
print("Step2 complete")
#Make predictions on the test set
y_pred = rf_regressor.predict(X_test)
print("Step3 complete")
#X_train.to_csv('X_train.csv', index=False)
#y_train.to_csv('y_train.csv', index=False)
#X_test.to_csv('X_test.csv', index=False)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

r2 = r2_score(y_test, y_pred)
print(f'R-squared Score: {r2}')

