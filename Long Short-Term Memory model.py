# Nowcasting model based on LSTM github of Hopp (also described in thesis) https://github.com/dhopp1/nowcast_lstm#nowcast_lstm

# In[1] inport libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dill # for loading and saving a trained model
from nowcast_lstm.LSTM import LSTM
from nowcast_lstm.model_selection import variable_selection, hyperparameter_tuning, select_model
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from sklearn.preprocessing import PowerTransformer
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)


# In[2] Importing the data (Data processing is similar to the file with the other models)
# For modeling without COVID-19, changes the excel file to "CompleteTSN (without COVID-19)" or adjust the dates for training and testing later in the file 

file_path = 'c:/Users/fmhob/OneDrive/Desktop/Tata/myenv/'
data = pd.read_excel(file_path + "CompleteTSN.xlsx", parse_dates=["Datum"])

data["date"] = pd.to_datetime(data["Datum"])

# For other products, replace Hot Rolled Narrow Strip
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


# In[3] List of features to test for stationarity

features = ["IP", "Manufacturing", "GVA automotive", "GVA construction", 
            "GVA machinery", "Number of cars produced in EU", "Number of cars produced in Germany", 
            "Manufacturing PMI - output", "Manufacturing PMI - new orders", "Manufacturing PMI - input prices", 
            "Manufacturing PMI - output prices", "Construction PMI", "Services PMI", 
            "EX ECONOMIC SENTIMENT INDICATOR VOLA", "EX IND.: OVERALL - INDL CONF INDICATOR SADJ", 
            "EX EC SERVICES SVY- TOTAL: SERVICES CONFIDENCE INDICATOR SADJ", "EX CONSUMER CONFIDENCE INDICATOR - EU SADJ", 
            "EX RET.: OVERALL - RET TRD CONF INDICATOR SADJ", "EX CNSTR.: OVERALL - CNSTR CONF INDICATOR SADJ", 
            "EX IND.: MV, TRAILERS & SEMI-TRAILERS - INDL CONF INDICATOR SADJ", "EX IND.: MACH & EQP NEC - INDL CONF INDICATOR SADJ", 
            "CPI", "PPI", "US$/EUR", "Ten-year bond", "Euribor"]

# Loop through the features, if not stationary apply differencing once and test again
for feature in features:
    result = adfuller(data[feature])
    p_value = result[1]
    
    if p_value < 0.05:
        print(f"{feature}: The time series is stationary.")
    else:
        print(f"{feature}: The time series is not stationary. Apply differencing to make it stationary.")
        
        # Apply differencing to make non-stationary series stationary
        data[feature] = data[feature].diff().fillna(0)

        # Retest for stationarity after differencing using ADF test
        result = adfuller(data[feature])
        p_value = result[1]

        if p_value < 0.05:
            print(f"{feature}: The time series is now stationary after differencing.")
        else:
            print(f"{feature}: Differencing did not make the series stationary.")

# Unemployment rate was not stationary after differencing once, try again
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


# In[4] Create a new data set with stationary data based on previous findings

stationary_data = data.copy()  # Create a copy of the original data

# List of features that needs to be differenced once
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

# Loop through these non-stationary features and apply differencing
for feature in non_stationary_features:
    stationary_data[feature] = stationary_data[feature].diff().fillna(0)


# In[4] Features that need to be differenced twice: unemployment rate and consumer confidence

# Copy the original data for the Unemployment rate
feature = "Unemployment rate"
unemployment_data = data[feature].copy()

# Apply differencing for 2 lags
for lag in range(3):
    unemployment_data = unemployment_data.diff().fillna(0)

# Add the modified Unemployment rate back to the stationary_data
stationary_data["Unemployment rate"] = unemployment_data 


# Copy the orignal data for the consumer confidence
feature2 = "EX CONSUMER CONFIDENCE INDICATOR - EU SADJ"
consumer_data = data[feature2].copy()

# Apply differencing for 2 lags
for lag in range(3):
    consumer_data = consumer_data.diff().fillna(0)

# Add the modified confidence rate back to the stationary_data
stationary_data["EX CONSUMER CONFIDENCE INDICATOR - EU SADJ"] = consumer_data 


# In[5] Replace "data" with the stationary data set
# To be sure, check stationary again using adfuller and kpss (note the difference in H0)

data = stationary_data
print(data)

for column in data.columns[2:]:
    result = adfuller(data[column])
    p_value = result[1]

    if p_value < 0.05:
        print(f"{column}: The time series is stationary.")
    else:
        print(f"{column}: The time series is not stationary.")
      

for column in data.columns[2:]:
    result = kpss(data[column], regression='c')
    p_value = result[1]

    if p_value > 0.05:
        print(f"{column}: The time series is stationary.")
    else:
        print(f"{column}: The time series is not stationary.")
     

# In[5] Transform and scale the data

# Select the features
columns_to_standardize = ["IP",
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
    "CPI", "PPI", "Unemployment rate", "US$/EUR", "Ten-year bond", "Euribor"]


#columns_to_standardize = ['EX IND.: MV, TRAILERS & SEMI-TRAILERS - INDL CONF INDICATOR SADJ', 'Number of Cars Produced in Europe', 'EX CNSTR.: OVERALL - CNSTR CONF INDICATOR SADJ', 'Manufacturing PMI - new orders', 'Manufacturing PMI - output', 'Number of cars produced in Germany', 'Unemployment rate', 'Euribor']

# Construct a pipleline with transformation and scaling (without train/test split)
#power_transformer = PowerTransformer(method = 'yeo-johnson', standardize=True)
#data[columns_to_standardize] = power_transformer.fit_transform(data[columns_to_standardize])

# Defining the train and test split, selecting the target variable

#train_end_date = "2020-12-01" # training data through X date
#training = data.loc[data['date'] <= train_end_date,:]
#target = "Hot Rolled Narrow Strip"

# Select the features) 
columns_to_standardize = ['Number of cars produced in EU', 'EX RET.: OVERALL - RET TRD CONF INDICATOR SADJ', 'Euribor', 'EX IND.: MACH & EQP NEC - INDL CONF INDICATOR SADJ', 'GVA construction', 'Manufacturing PMI - input prices', 'PPI', 'GVA machinery']

# In[6] Construct a pipleline with transformation and scaling:

# The lstm_nowcast library does not allow for separate input of the train and test set;
# It handles the splitting and validation internally.
# Therefore, all data transformations should be done beforehand.

# To minimize "spillover effects," I split the data based on the end date of the training set.
# Then I apply the pipeline and concatenate the data to create a unified dataset.
power_transformer = PowerTransformer(method = 'yeo-johnson', standardize=True)

train_end_date = "2021-01-01" # the end data is set on December 17 for the test without Covid-19, January 2021 with covid. (Keep the last 2 years for testing the model)

# Split the data into train and test sets
train_data = data[data['date'] <= train_end_date].copy()
test_data = data[data['date'] > train_end_date].copy()  # Use .copy() to avoid SettingWithCopyWarning

# Apply transformations to the training set
train_data.loc[:, columns_to_standardize] = power_transformer.fit_transform(train_data[columns_to_standardize])

# Apply transformations to the test set using the parameters from the training set
test_data.loc[:, columns_to_standardize] = power_transformer.transform(test_data[columns_to_standardize])

# Concatigate both sets
data = pd.concat([train_data, test_data], axis=0)

training = data.loc[data['date'] <= train_end_date,:]
target = "Hot Rolled Narrow Strip"


# In[7]

# Fit a (first) LSTM model on the total  data
model = LSTM(training, target, n_timesteps=12, n_models=10)
model.train(quiet=True)
print(model.feature_contribution())

test_preds = model.predict(data, only_actuals_obs=True).loc[lambda x: x.date > train_end_date,:] # passing the full dataset, then filtering only for predictions on dates the model wasn't trained on

# Test the predictions and calculate the RMSE
# rmse = np.sqrt(np.nanmean((test_preds.predictions - test_preds.actuals)**2))

# Plot de actuals and predictions
# plt.plot(test_preds.date, test_preds.actuals, label="actual") 
# plt.plot(test_preds.date, test_preds.predictions, label="predictions")
# plt.legend()
# plt.title(f"RMSE: {rmse}");


# In[8]

# Predicting the 99% and 95% confidence interval
# interval_preds_99 = model.interval_predict(data, interval=0.99, only_actuals_obs=True, start_date="2021-06-01", end_date="2023-04-01")
# interval_preds_95 = model.interval_predict(data, interval=0.95, only_actuals_obs=True, start_date="2021-06-01", end_date="2023-04-01")
#interval_preds_50 = model.interval_predict(data, interval=0.5, only_actuals_obs=True, start_date="2020-12-01", end_date="2023-01-01")

# plotting the results 
# fig, ax = plt.subplots()
# ax.set_title('LSTM Prediction of Lengths Cut From Hot Rolled Wide')
# ax.tick_params(axis='x', rotation=45)
# ax.plot(interval_preds_99.date,interval_preds_99.actuals, label="actuals")
# ax.plot(interval_preds_99.date,interval_preds_99.predictions, label="preds")
# ax.fill_between(interval_preds_99.date, interval_preds_99.lower_interval, interval_preds_99.upper_interval, color='b', alpha=.1, label="0.99 interval")
# ax.fill_between(interval_preds_95.date, interval_preds_95.lower_interval, interval_preds_95.upper_interval, color='r', alpha=.1, label="0.95 interval")
#ax.fill_between(interval_preds_50.date, interval_preds_50.lower_interval, interval_preds_50.upper_interval, color='grey', alpha=.1, label="0.5 interval")
# plt.legend()


# In[8] Automatic function to test on different hyperparameters and features (LSTM specific feature selection, runs differnt sets of features with differnt sets of parameters)

#selection_results = select_model(training, target, n_models=10, n_timesteps_grid = [9, 12], 
#                    train_episodes_grid = [100, 200], batch_size_grid = [30], n_hidden_grid = [10, 20],           
#                    n_layers_grid = [2], n_folds=5, init_test_size = 0.5, 
#                    initial_ordering="univariate")

# In[]
#print(selection_results)
#print(selection_results.iloc[0,0])
#print(selection_results.iloc[0,1])
#print(selection_results.iloc[1,0])
#print(selection_results.iloc[2,0])

# In[9] Automatic function to test on different hyperparameters given a fixed set of features
target = "Hot Rolled Narrow Strip"
#tuning_results = hyperparameter_tuning(training, target, n_models=10, n_timesteps_grid = [18, 24], train_episodes_grid = [400, 800], batch_size_grid = [20,40], n_hidden_grid = [10, 40], n_layers_grid = [2], n_folds=2, init_test_size = 0.3)

# In[] 
#print(tuning_results)
#print(tuning_results.iloc[0,0])
#print(tuning_results.iloc[0,1])

# In[]
# performance on the test set with selected hyperparameters and variables
model = LSTM(training.loc[:, ["date", target] + ['Number of cars produced in EU', 'PPI', 'EX IND.: MACH & EQP NEC - INDL CONF INDICATOR SADJ', 'GVA machinery', 'CPI', 'EX CONSUMER CONFIDENCE INDICATOR - EU SADJ', 'Unemployment rate', 'Manufacturing PMI - input prices']], target, n_timesteps=24, n_models=10, train_episodes = 2000, batch_size = 20, n_hidden = 40, n_layers = 2, decay=0.98)
model.train(quiet=True)

test_preds = model.predict(data.loc[:, ["date", target] + ['Number of cars produced in EU', 'PPI', 'EX IND.: MACH & EQP NEC - INDL CONF INDICATOR SADJ', 'GVA machinery', 'CPI', 'EX CONSUMER CONFIDENCE INDICATOR - EU SADJ', 'Unemployment rate', 'Manufacturing PMI - input prices']], only_actuals_obs=True).loc[lambda x: x.date > train_end_date,:] # passing the full dataset, then filtering only for predictions on dates the model wasn't trained on

# rmse on the test set of the model
rmse = np.sqrt(np.nanmean((test_preds.predictions - test_preds.actuals)**2))

# Plotting the results
plt.plot(test_preds.date, test_preds.actuals, label="actual") 
plt.plot(test_preds.date, test_preds.predictions, label="predictions")
plt.legend()
plt.title(f"RMSE: {rmse}");

# In[10] Obtaining the intervals of the improved model
print(data.columns)


# In[11]
# Predicting the 99% and 95% confidence interval
#interval_preds_99 = model.interval_predict(data.loc[:, ["date", target] + ['Number of cars produced in EU', 'EX RET.: OVERALL - RET TRD CONF INDICATOR SADJ', 'Euribor', 'EX IND.: MACH & EQP NEC - INDL CONF INDICATOR SADJ', 'GVA construction', 'Manufacturing PMI - input prices', 'PPI', 'GVA machinery']], interval=0.99, only_actuals_obs=True, start_date="2021-01-01", end_date="2023-01-01")
#interval_preds_95 = model.interval_predict(data.loc[:, ["date", target] + ['Number of cars produced in EU', 'EX RET.: OVERALL - RET TRD CONF INDICATOR SADJ', 'Euribor', 'EX IND.: MACH & EQP NEC - INDL CONF INDICATOR SADJ', 'GVA construction', 'Manufacturing PMI - input prices', 'PPI', 'GVA machinery']], interval=0.95, only_actuals_obs=True, start_date="2021-01-01", end_date="2023-01-01")
interval_preds_99 = model.interval_predict(data.loc[:, ["date", target] + ['Number of cars produced in EU', 'PPI', 'EX IND.: MACH & EQP NEC - INDL CONF INDICATOR SADJ', 'GVA machinery', 'CPI', 'EX CONSUMER CONFIDENCE INDICATOR - EU SADJ', 'Unemployment rate', 'Manufacturing PMI - input prices']], interval=0.99, only_actuals_obs=True, start_date="2021-01-01", end_date="2023-01-01") # From Januari 2021 or March 2018 ex Covic-19)
interval_preds_95 = model.interval_predict(data.loc[:, ["date", target] + ['Number of cars produced in EU', 'PPI', 'EX IND.: MACH & EQP NEC - INDL CONF INDICATOR SADJ', 'GVA machinery', 'CPI', 'EX CONSUMER CONFIDENCE INDICATOR - EU SADJ', 'Unemployment rate', 'Manufacturing PMI - input prices']], interval=0.95, only_actuals_obs=True, start_date="2021-01-01", end_date="2023-01-01")
#interval_preds_50 = model.interval_predict(data, interval=0.5, only_actuals_obs=True, start_date="2020-12-01", end_date="2023-01-01")

# In[12]
# plotting the results 
fig, ax = plt.subplots()
ax.set_title('LSTM Prediction of Hot Rolled Narrow Strip')
ax.tick_params(axis='x', rotation=45)
ax.plot(interval_preds_99.date,interval_preds_99.actuals, label="Data")
ax.plot(interval_preds_99.date,interval_preds_99.predictions, label="LSTM predictions")
ax.fill_between(interval_preds_99.date, interval_preds_99.lower_interval, interval_preds_99.upper_interval, color='b', alpha=.1, label="0.99 interval")
ax.fill_between(interval_preds_95.date, interval_preds_95.lower_interval, interval_preds_95.upper_interval, color='r', alpha=.1, label="0.95 interval")
plt.ylabel('Apparent Steel Demand (Metric Tons)')

#ax.fill_between(interval_preds_50.date, interval_preds_50.lower_interval, interval_preds_50.upper_interval, color='grey', alpha=.1, label="0.5 interval")
plt.legend()

# %%
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
# Calculate metrics for LSTM model
rmse_lstm = np.sqrt(np.nanmean((test_preds.predictions - test_preds.actuals)**2))
mae_lstm = mean_absolute_error(test_preds.actuals, test_preds.predictions)
mpe_lstm = mean_absolute_percentage_error(test_preds.actuals, test_preds.predictions)

# Print the results
print("LSTM Model Metrics:")
print("Root Mean Squared Error (RMSE):", rmse_lstm)
print("Mean Absolute Error (MAE):", mae_lstm)
print("Mean Percentage Error (MPE):", mpe_lstm)

# %%
