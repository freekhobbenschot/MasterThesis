import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import concatenate
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # for PCA calculation
import matplotlib.pyplot as plt # for plotting
import matplotlib.patches as mpatches
from sklearn.preprocessing import PowerTransformer



# Reading in data
file_path = 'c:/Users/fmhob/OneDrive/Desktop/Tata/'

# Load and preprocess the data
df = pd.read_excel(
    file_path + "CompleteTSN.xlsx",
    usecols="H:AH",  # Adjust columns as needed, e.g., "A:AH" for all columns
    sheet_name="Data",
    #index_col='Datum',
    parse_dates=True
)

df2 = pd.read_excel(
    file_path + "CompleteTSN.xlsx",
    usecols="B",  # Adjust columns as needed, e.g., "A:AH" for all columns
    sheet_name="Data",
    #index_col='Datum',
    parse_dates=True
)

column_data_types = df.dtypes
print(column_data_types)

print(df.shape)
df = df.fillna(0)

power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
X_std = power_transformer.fit_transform(df)

pca = PCA()
X_pca = pca.fit(X_std)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")
plt.show()

num_components = 15
pca = PCA(num_components)
X_pca = pca.fit_transform(X_std) # fit and reduce dimension

pca_components_df = pd.DataFrame(pca.components_, columns = df.columns)
print(pca_components_df)

n_pcs = pca.n_components_ # get number of component

# get the index of the most important feature on EACH component

most_important = [np.abs(pca.components_[i]).argmax() for i in
range(n_pcs)]

intial_feature_names = df.columns

# get the most important feature names

most_important_feature_names = [intial_feature_names[most_important[i]]
for i in range(n_pcs)]
print(most_important_feature_names)


###################################################################33
# Incremental PCA and other parts not used for the feature selection
# Incremental PCA is mostly used for large data sets (for example when there are memory issues)

# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel("number of components")
# plt.ylabel("cumulative explained variance")
# plt.show()

## IPCA Evaluation
#from sklearn.decomposition import IncrementalPCA

#ipca = IncrementalPCA(n_components = 4, batch_size=10)
#X_ipca = ipca.fit_transform(X_std)
#print(ipca.n_components_)

#principle_components_df = pd.DataFrame(ipca.components_, columns = df.columns)
#print(principle_components_df)

#n_pcs_ipca = ipca.n_components_
#most_important_ipca = [np.abs(ipca.components_[i]).argmax() for i in range(n_pcs_ipca)]
#initial_feature_names = df.columns

#get most important feature names
#most_important_feature_names_ipca = [initial_feature_names[most_important_ipca[i]]
#for i in range(n_pcs_ipca)]

#print(most_important_feature_names_ipca)

#MSDA
#import operator, statistics
#from statistics import mean
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
#from msda.msda import *

#print("Trend information for each column in the dataset")
#print("Dictionary of each column with variation count\n", df.apply(FeatureSelection.count_trend, axis=0))

#def count_trend(column):
#    inc_count = (column.diff() > 0).sum()  # Count of increases
#    dec_count = (column.diff() < 0).sum()  # Count of decreases
#    eql_count = (column.diff() == 0).sum()  # Count of equals
#    return {'Inc': inc_count, 'Dec': dec_count, 'Eql': eql_count}

# Create a dictionary of trend information for each column
# trend_info = {}
# for col in df.columns:
#    trend_info[col] = count_trend(df[col])

# Find the column with maximum variation in each category (Inc, Dec, Eql)
#max_variations = {
#    'Inc': max(trend_info.items(), key=lambda x: x[1]['Inc'])[0],
#    'Dec': max(trend_info.items(), key=lambda x: x[1]['Dec'])[0],
#    'Eql': max(trend_info.items(), key=lambda x: x[1]['Eql'])[0]
#}

#print('Max. Variation Involved in each Sensor Column values are:')
#print('Note: Inc-Increasing ; Dec-Decreasing ; Eq-Equal ')
#print(f'For Increasing Variations: {max_variations["Inc"]}')
#print(f'For Decreasing Variations: {max_variations["Dec"]}')
#print(f'For Equal Variations: {max_variations["Eql"]}')

###


import numpy as np
from sklearn.decomposition import FastICA

# Define the desired number of components
L = 10

# Create an ICA instance
ica = FastICA(n_components=L, max_iter=200)

# Fit ICA to your input data (assuming X_std is your data matrix)
Y = ica.fit_transform(X_std)

# Get the mixing matrix (W matrix)
mixing_matrix = ica.mixing_

# Print or inspect the mixing matrix to understand how features contribute to the components
print("Mixing Matrix:")
print(mixing_matrix)

# Calculate the correlation between each feature and the outcome (assumes X_std and df2 are defined)
correlations = np.abs(np.corrcoef(X_std, df2, rowvar=False)[:-1, -1])

# Sort the correlations in descending order and keep track of column indices
sorted_indices = np.argsort(correlations)[::-1]

# Select the top L features
selected_features = X_std[:, sorted_indices[:L]]
print("Selected Features:")
print(selected_features)

# Get the most important feature names (assuming they are stored in df.columns)
most_important = [sorted_indices[i] for i in range(L)]
initial_feature_names = df.columns
most_important_feature_names = [initial_feature_names[most_important[i]] for i in range(L)]

print("Most Important Feature Names:")
print(most_important_feature_names)
