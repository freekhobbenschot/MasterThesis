#I used the following github to brainstorm for ideas on how to program the models: https://github.com/OrestisMk/OrestisMk-Multivariate-forecast-with-VAR-SVR-RNN-LSTM/blob/main/RNN-LSTM(2%20models).ipynb


# In[1] Import necessary libraries
import tensorflow as tf
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
import dill
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PowerTransformer
from nowcast_lstm.LSTM import LSTM
from nowcast_lstm.model_selection import variable_selection, hyperparameter_tuning, select_model
import warnings

warnings.filterwarnings("ignore")
np.random.seed(1234)

# In[2] Import the data, select the target variable

#Complete Data (CompleteTSN file)
file_path = 'c:/Users/fmhob/OneDrive/Desktop/Tata/'
data = pd.read_excel(file_path + "CompleteTSN.xlsx", parse_dates=["date"])

#Data without Covid period (CompleteTSN - without COVID-19 file)
#file_path = 'c:/Users/fmhob/OneDrive/Desktop/Tata/myenv/'
#data = pd.read_excel(file_path + "CompleteTSN.xlsx (without COVID-19)", parse_dates=["Datum"])
data["date"] = pd.to_datetime(data["Datum"])


data = data.loc[:, ["date", "Hot Rolled Narrow Strip", "IP",
    "Manufacturing", "GVA automotive", "GVA construction", "GVA machinery",
    "Number of cars produced in EU", "Number of cars produced in Germany",
    "Manufacturing PMI - output", "Manufacturing PMI - new orders",
    "Manufacturing PMI - input prices", "Manufacturing PMI - output prices",
    "Construction PMI", "Services PMI", "EX ECONOMIC SENTIMENT INDICATOR VOLA",
    "EX IND.: OVERALL - INDL CONF INDICATOR SADJ",
    "EX EC SERVICES SVY- TOTAL: SERVICES CONFIDENCE INDICATOR SADJ",
    "EX CONSUMER CONFIDENCE INDICATOR - EU SADJ",
    "EX RET.: OVERALL - RET TRD CONF INDICATOR SADJ",
    "EX CNSTR.: OVERALL - CNSTR CONF INDICATOR SADJ",
    "EX IND.: MV, TRAILERS & SEMI-TRAILERS - INDL CONF INDICATOR SADJ",
    "EX IND.: MACH & EQP NEC - INDL CONF INDICATOR SADJ",
    "CPI", "PPI", "Unemployment rate", "US$/EUR", "Ten-year bond", "Euribor"]] # random subset of columns for simplicity


# In[3] Test for stationarity (data processing is similar as for LSTM)

# List of features to test for stationarity
features = ["IP", "Manufacturing", "GVA automotive", "GVA construction", 
            "GVA machinery", "Number of cars produced in EU", "Number of cars produced in Germany", 
            "Manufacturing PMI - output", "Manufacturing PMI - new orders", "Manufacturing PMI - input prices", 
            "Manufacturing PMI - output prices", "Construction PMI", "Services PMI", 
            "EX ECONOMIC SENTIMENT INDICATOR VOLA", "EX IND.: OVERALL - INDL CONF INDICATOR SADJ", 
            "EX EC SERVICES SVY- TOTAL: SERVICES CONFIDENCE INDICATOR SADJ", "EX CONSUMER CONFIDENCE INDICATOR - EU SADJ", 
            "EX RET.: OVERALL - RET TRD CONF INDICATOR SADJ", "EX CNSTR.: OVERALL - CNSTR CONF INDICATOR SADJ", 
            "EX IND.: MV, TRAILERS & SEMI-TRAILERS - INDL CONF INDICATOR SADJ", "EX IND.: MACH & EQP NEC - INDL CONF INDICATOR SADJ", 
            "CPI", "PPI", "US$/EUR", "Ten-year bond", "Euribor"]

# Loop through the features and perform ADF test
for feature in features:
    result = adfuller(data[feature])
    p_value = result[1]
    
    if p_value < 0.05:
        print(f"{feature}: The time series is stationary.")
    else:
        print(f"{feature}: The time series is not stationary. Applying differencing to make it stationary.")
        
        # Apply differencing to make non-stationary series stationary
        data[feature] = data[feature].diff().fillna(0)

        # Retest for stationarity after differencing using ADF test
        result = adfuller(data[feature])
        p_value = result[1]

        if p_value < 0.05:
            print(f"{feature}: The time series is now stationary after differencing.")
        else:
            print(f"{feature}: Differencing did not make the series stationary.")

# ADF Test for Unemployment rate with varying lags
feature = "Unemployment rate"

for num_lags in range(1, 3):
    # Copy the original data for each test
    test_data = data[feature].copy()
    
    # Apply differencing for the specified number of lags
    for lag in range(num_lags):
        test_data = test_data.diff().fillna(0)
    
    # Perform ADF Test
    result = adfuller(test_data)
    p_value = result[1]
    
    if p_value < 0.05:
        print(f"{feature} with {num_lags} lags: The time series is now stationary.")
    else:
        print(f"{feature} with {num_lags} lags: The time series is not stationary.")


# In[4] Make data stationary

# Create a new DataFrame for stationary data
stationary_data = data.copy()  # Create a copy of the original data

# List of features that are not stationary
non_stationary_features = [
    "IP", "Manufacturing", "GVA construction", "GVA machinery", "GVA automotive",
    "Manufacturing PMI - output prices", "Construction PMI", "EX IND.: MACH & EQP NEC - INDL CONF INDICATOR SADJ",
    "Number of cars produced in EU", "Number of cars produced in Germany", 
    "EX ECONOMIC SENTIMENT INDICATOR VOLA",
    "EX IND.: OVERALL - INDL CONF INDICATOR SADJ",
    "EX EC SERVICES SVY- TOTAL: SERVICES CONFIDENCE INDICATOR SADJ",
    "EX CONSUMER CONFIDENCE INDICATOR - EU SADJ",
    "EX RET.: OVERALL - RET TRD CONF INDICATOR SADJ",
    "EX CNSTR.: OVERALL - CNSTR CONF INDICATOR SADJ",
    "EX IND.: MV, TRAILERS & SEMI-TRAILERS - INDL CONF INDICATOR SADJ",
    "CPI", "PPI", "US$/EUR", "Ten-year bond", "Euribor"
]

# Loop through the non-stationary features and apply differencing
for feature in non_stationary_features:
    stationary_data[feature] = stationary_data[feature].diff().fillna(0)



#In[5] Additional differing for the unemployment rate and consumer confidence

# ADF Test for Unemployment rate with 2 lags
feature = "Unemployment rate"

# Copy the original data for the Unemployment rate
unemployment_data = data[feature].copy()

# Apply differencing for 2 lags
for lag in range(3):
    unemployment_data = unemployment_data.diff().fillna(0)

# Add the modified Unemployment rate back to the stationary_data
stationary_data["Unemployment rate"] = unemployment_data 


feature2 = "EX CONSUMER CONFIDENCE INDICATOR - EU SADJ"
# Copy the original data for the confidence indicator
confidence_data = data[feature2].copy()

# Apply differencing for 2 lags
for lag in range(3):
    confidence_data = confidence_data.diff().fillna(0)

# Add the modified Unemployment rate back to the stationary_data
stationary_data['EX CONSUMER CONFIDENCE INDICATOR - EU SADJ'] = confidence_data 

# In[20]
df2 = data['date']
print(df2)

# In[6] Chose the data that will be used for analysis 

selected_columns = ['Hot Rolled Narrow Strip', 'Number of cars produced in EU', 'PPI', 'EX IND.: MACH & EQP NEC - INDL CONF INDICATOR SADJ', 'GVA machinery', 'CPI', 'EX CONSUMER CONFIDENCE INDICATOR - EU SADJ', 'Unemployment rate', 'Manufacturing PMI - input prices'] 

stationary_data = stationary_data[selected_columns]
stationary_data.set_index(stationary_data.index, inplace=True)  # Use existing index as the new index

print("DataFrame Shape: {} rows, {} columns".format(stationary_data.shape[0], stationary_data.shape[1])) 
print(stationary_data.head())

df = stationary_data
print(df.isnull().any())
print(df)



# In[7] Split the target variable from the features

X1 = df.iloc[:, 1:6].values  # Assuming features start from the 2nd column (index 1)
y1 = df.iloc[:, 0].values  # Assuming the target variable is in the 1st column (index 0)

print(X1)
print(y1)
print(y1.shape)

#In[8] MIN MAX SCALING (Did not use this for the final results)

#from sklearn.preprocessing import MinMaxScaler
#sc = MinMaxScaler()
#sc_X = MinMaxScaler()
#sc_y = MinMaxScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)
#y_train = sc_y.fit_transform(y_train.reshape(-1,1))
#y_test = sc_y.transform(y_test.reshape(-1,1))

# from sklearn.model_selection import TimeSeriesSplit

# tscv = TimeSeriesSplit(n_splits=5)  # You can adjust the number of splits

#for train_index, test_index in tscv.split(X1):
#    X_train, X_test = X1[train_index], X1[test_index]
#    y_train, y_test = y1[train_index], y1[test_index]
#print(X_train)
#print(X_test)
#print(y_train)
#print(y_test)

#from sklearn.preprocessing import PowerTransformer
#data = df 

#columns_to_standardize = ["IP",
#                    "Manufacturing", "GVA automotive", "GVA construction", "GVA machinery",
#                    "Number of cars produced in EU", "Number of cars produced in Germany",
#                    "Manufacturing PMI - output"]

#power_transformer = PowerTransformer(method = 'yeo-johnson', standardize=True)
#data[columns_to_standardize] = power_transformer.fit_transform(data[columns_to_standardize])

#print(data)

#In[8] Split the data and apply transformation and scaling

tscv = TimeSeriesSplit(n_splits=5)  # Number of splits, here I chose 5 to have 20% for test data

for train_index, test_index in tscv.split(X1):
    X_train, X_test = X1[train_index], X1[test_index]
    y_train, y_test = y1[train_index], y1[test_index]

from sklearn.preprocessing import PowerTransformer

columns_to_standardize = ['Number of cars produced in EU', 'PPI', 'EX IND.: MACH & EQP NEC - INDL CONF INDICATOR SADJ', 'GVA machinery', 'CPI', 'EX CONSUMER CONFIDENCE INDICATOR - EU SADJ', 'Unemployment rate', 'Manufacturing PMI - input prices']
power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)

# Transform and standardize X_train
X_train = power_transformer.fit_transform(X_train)
# Transform and standardize X_test using the parameters from X_train
X_test = power_transformer.transform(X_test)


#In[9] Import the libraries that are necessary for modeling

from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import make_scorer


#In[10] Determine the best combination of parameters 

param_grid = {'C': [10, 100, 1000, 2000, 3000],
              'gamma': [0.0001, 0.001, 0.01, 0.1],
              'kernel': ['rbf'],
              'epsilon': [1e-5, 1e-3, 1e-2]}
scorer = make_scorer(mean_squared_error, greater_is_better=False)
model = SVR()
grid = GridSearchCV(SVR(), param_grid, cv=10, scoring=scorer, refit=True, verbose=1)
grid.fit(X_train, np.ravel(y_train))

# Print results
print("Best parameters:", grid.best_params_)
print("Best estimator:", grid.best_estimator_)
print("Best score:", grid.best_score_)

cv_scores = cross_val_score(SVR(kernel='rbf', C=10, epsilon=0.001, gamma=1), X_train, np.ravel(y_train), cv=10, scoring=scorer)
print("Cross-Validation Scores:", cv_scores)


#In[11] Train and test the model with the best parameter selection

reg = SVR(kernel = 'rbf', C = 1000 ,epsilon = 0.01, gamma = 0.0001)
reg = reg.fit(X_train, y_train)
grid_predictions = grid.predict(X_test)
grid_predictions.reshape(1,-1)


#In[12] Definine the performance metrics 

from sklearn.metrics import r2_score

r2 = r2_score(y_test, grid_predictions)
print("R-squared Score:", r2)
rmse = np.sqrt(mean_squared_error(y_test,grid.predict(X_test)))
print('RMSE value of the SVR Model is:', (1-rmse)*100)
mape =np.mean(np.abs((y_test-grid_predictions)/y_test))*100
print('MAPE value of the SVR Model is:', (mape))
mpe = np.mean(y_test-grid_predictions)*100
print('MPE value of the SVR Model is:', (mpe))



#In[12] Plot predictions and actuals 

# Create a time index for the test data
time_index = data.index[-len(y_test):]
print(time_index)

# Plot the unnormalized true values and SVR predictions
plt.figure(figsize=(10, 6))
plt.plot(time_index, y_test, label='True Values', color='blue')
plt.plot(time_index, grid_predictions, label='SVR Predictions', color='red')
plt.title('SVR Predictions vs. True Values')
plt.xlabel('Time')
plt.ylabel('Apparent Steel Demand (Metric Tons)')
plt.legend()
plt.grid(True)
plt.show()


# Print true values and predictions for verification
# print("True Values:", y_test)
# print("SVR Predictions:", grid_predictions)


#In[13] Continue with the Random Forest model

from sklearn.ensemble import RandomForestRegressor
from keras.utils import plot_model


# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a RandomForestRegressor
rf_model = RandomForestRegressor(random_state=42)

# Use GridSearchCV to search for the best parameters
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the model with the best parameters
rf_model = RandomForestRegressor(random_state=42, **best_params)
rf_model.fit(X_train, y_train)

# Make predictions
svr_predictions = grid_predictions
rf_predictions = rf_model.predict(X_test)

# Create an ensemble model based on RF and SVR (simple average)
ensemble_predictions = (svr_predictions + rf_predictions) / 2


#In[19]
# Plot the models together
plt.figure(figsize=(10, 6))
plt.plot(data.index[-len(y_test):], svr_predictions, label='SVR Predictions', color='blue')
plt.plot(data.index[-len(y_test):], rf_predictions, label='Random Forest Predictions', color='green')
plt.plot(data.index[-len(y_test):], ensemble_predictions, label='Ensemble Predictions', color='red')
plt.plot(data.index[-len(y_test):], y_test, label='True Values', color='black', linestyle='--')
plt.title('SVR, Random Forest, and Ensemble Predictions vs. True Values')
plt.xlabel(data.index[-len(y_test):])
plt.ylabel('Apparent Steel Demand (Metric Tons)')
plt.legend()
plt.grid(True)
plt.show()


# In[19] Compare the last months
# svr_predictions = svr_predictions[4:]
# rf_predictions = rf_predictions[4:]
# ensemble_predictions = ensemble_predictions[4:]
# y_test = y_test[4:]


# In[] 
# Plot the models together (Include dates on the x-axis) 
plt.figure(figsize=(10, 6))
plt.xticks(rotation=34)
plt.plot(df2[-len(y_test-5):],svr_predictions, label='SVR Predictions', color='blue')
plt.plot(df2[-len(y_test-5):],rf_predictions, label='Random Forest Predictions', color='green')
plt.plot(df2[-len(y_test-5):],ensemble_predictions, label='Ensemble Predictions', color='red')
plt.plot(df2[-len(y_test-5):],y_test, label='Data', color='black', linestyle='--')
plt.title('SVR, Random Forest, Ensemble Predictions and Data for Hot Rolled Narrow Strip')
plt.ylabel('Apparent Steel Demand (Metric Tons)')
plt.legend()
plt.grid(True)
plt.show()

# In[20]


# Plot the models together 
# Remove the last 6 months due to expected revisions in the data

# plt.figure(figsize=(10, 6))
# plt.xticks(rotation=34)
# plt.plot(df2[-30:-5],svr_predictions[:-5], label='SVR Predictions', color='blue')
# plt.plot(df2[-30:-5],rf_predictions[:-5], label='Random Forest Predictions', color='green')
# plt.plot(df2[-30:-5],ensemble_predictions[:-5], label='Ensemble Predictions', color='red')
# plt.plot(df2[-30:-5],y_test[:-5], label='Data', color='black', linestyle='--')
# plt.title('SVR, Random Forest, Ensemble Predictions and Data for Hot Rolled Narrow Strip')
# plt.ylabel('Apparent Steel Demand (Metric Tons)')
# plt.legend()
# plt.grid(True)
# plt.show()

# Calculate Mean Squared Error for SVR
# mse_svr = mean_squared_error(y_test, svr_predictions)
# print("Mean Squared Error (SVR):", mse_svr)

# Calculate Mean Squared Error for RandomForestRegressor
# mse_rf = mean_squared_error(y_test, rf_predictions)
# print("Mean Squared Error (Random Forest):", mse_rf)

# mse_ens = mean_squared_error(y_test, ensemble_predictions)
# print("Mean Squared Error (Ensemble_model):", mse_ens)




# In[14] Calculating the results for each of the models

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

# Calculate Mean Squared Error (RMSE)
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Calculate Mean Percentage Error (MPE)
def mean_percentage_error(y_true, y_pred):
    return np.mean((y_true - y_pred) / y_true) * 100

# Calculate metrics for SVR
rmse_svr = root_mean_squared_error(y_test, svr_predictions)
mae_svr = mean_absolute_error(y_test, svr_predictions)
mpe_svr = mean_absolute_percentage_error(y_test, svr_predictions)


# Calculate metrics for RandomForestRegressor
rmse_rf = root_mean_squared_error(y_test, rf_predictions)
mae_rf = mean_absolute_error(y_test, rf_predictions)
mpe_rf = mean_absolute_percentage_error(y_test, rf_predictions)


# Calculate metrics for Ensemble model
rmse_ens = root_mean_squared_error(y_test, ensemble_predictions)
mae_ens = mean_absolute_error(y_test, ensemble_predictions)
mpe_ens = mean_absolute_percentage_error(y_test, ensemble_predictions)

# Print the results
print("SVR Metrics:")
print("Root Mean Squared Error (RMSE):", rmse_svr)
print("Mean Absolute Error (MAE):", mae_svr)
print("Mean Absolute Percentage Error (MPE):", mpe_svr)
print()

print("Random Forest Metrics:")
print("Root Mean Squared Error (RMSE):", rmse_rf)
print("Mean Absolute Error (MAE):", mae_rf)
print("Mean Absolute Percentage Error (MPE):", mpe_rf)
print()

print("Ensemble Model Metrics:")
print("Root Mean Squared Error (RMSE):", rmse_ens)
print("Mean Absolute Error (MAE):", mae_ens)
print("Mean Absolute Percentage Error (MPE):", mpe_ens)



# Extract month names for the x-axis
months = data.index[-len(y_test)]  # Use strftime to get the month name

# %%
print(months)
# %%
import matplotlib.dates as mdates

# ...
# ...

# Plot the models together
plt.figure(figsize=(10, 6))
plt.plot(time_index, y_test, label='True Values', color='blue')
plt.plot(time_index, grid_predictions, label='SVR Predictions', color='red')
plt.title('SVR Predictions vs. True Values')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

# Set x-axis labels with correct dates
date_labels = [date.strftime('%b %Y') for date in time_index]
plt.xticks(time_index, date_labels, rotation=45, ha='right')  # Adjust rotation for better readability

plt.grid(True)
plt.show()

# %%
