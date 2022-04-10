#======================================#
#====    02 Feature Engineering    ====#
#======================================#

# 01 Data Split
# 02 Missing Values
# 03 Scaling data
# 04 Code Wind Direction
# 05 Test Dataset
# 06 End Engineering


# %%

#----    Settings    ----#

import sys, os
import shelve
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'mycode'))

import numpy as np
import pandas as pd
from scipy import stats 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# from mycode/
import utils
import myPlots 

# get data

d_old = shelve.open(os.path.join(os.path.dirname(sys.path[0]),'outputs', '01_data_explore'))
df = d_old['df'].copy()
d_old.close()

#%%
#----    01 Data Split    ----#

# Check availabe years
print(utils.my_count(df['Year']))

df['Set'] = pd.cut(df['Year'], bins=[2006, 2015, 2017], 
                   labels=['Training', 'Test'])
utils.my_count(df['Set'])

# %%
# Get train and test data
df_train = df[df['Set'] == 'Training'].copy()
df_test = df[df['Set'] == 'Test'].copy()

# %% 

# Update diff in dates as we removed dates after 2015-12-31 
# Failing if looking for values after 2015-12-31
df_train['DiffPrev'] = utils.get_days_diff(df_train, periods = 1)
df_train['DiffNext'] = utils.get_days_diff(df_train, periods = -1)

# %%
# Check representativeness test data
# Numeric columns
kk =df_test.describe() - df_train.describe()

# %%
# Categorical columns
for col_name in df_test.select_dtypes(include=['object']).columns:
    print(df_test[col_name].value_counts(dropna = False, normalize = True) -\
        df_train[col_name].value_counts(dropna = False, normalize = True))

# %%
# Plot descriptive on train data
col_to_plot = df_train.select_dtypes(include=np.number)\
    .columns.drop(['RainTomorrow01', 'DiffPrev', 'DiffNext'])

# Numeric columns
myPlots.plot_descriptive(df_train, 
                         columns = col_to_plot, 
                         outcome="RainTomorrow") 

# %%

# Check relation Today and Tomorrow
contingency_rain = df_train.groupby(['RainToday', 'RainTomorrow']).size().reset_index(name='counts')

tot_group_rain = contingency_rain.groupby(['RainToday'])['counts'].sum()

contingency_rain = contingency_rain.merge(tot_group_rain, on = "RainToday")
contingency_rain['counts'] = contingency_rain['counts_x']
contingency_rain['propToday'] = contingency_rain['counts_x'] /contingency_rain['counts_y']
contingency_rain.drop(['counts_x', 'counts_y'],  axis=1, inplace=True)
contingency_rain
#%%

#----    02 Missing Values    ----#

# Set 'Missing' for missing categorical variables

object_columns = df_train.select_dtypes(include = ['object']).columns
df_train[object_columns] = df_train[object_columns].fillna('Missing')

df_train.describe(include = ['object'])

# %%
# NaN proportion numerical variables

numeric_columns = df_train.select_dtypes(include=np.number).columns
df_train[numeric_columns].isnull().sum() / len(df_train)

# %%
# Save trimmed mean value or median within each location (before substitute NaN)
cols_to_median = ['Rainfall', 'Evaporation', 'RainfallLog', 'EvaporationLog'] # skewed distributed
cols_to_mean = [colname for colname in numeric_columns if \
    colname not in cols_to_median]

# Get trimmed mean values
trimmed_means_val = df_train.groupby('Location')[cols_to_mean].aggregate(
    lambda x: stats.trim_mean(x[x.notna()], proportiontocut=.05))

# Get median values
median_val = df_train.groupby('Location')[cols_to_median].aggregate(np.nanmedian)

reference_values = pd.concat([trimmed_means_val, median_val], axis=1)

# If no data for location is available, set the mean of trimmed means or medians
reference_values = reference_values.apply(lambda x: x.fillna(x.mean()), axis=0)
reference_values

# %%
# Substitute NaN with the mean yesterday and tomorrow values if possible 
cols_smart_fillna = ['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm', 'TempDiff', 'MaxMinDiff']
for col_name in cols_smart_fillna:
    utils.smart_fillna(df_train, col_name)

# %%
# Substitute remaining NAN with trimmed mean value
for col_name in reference_values.columns:
    var_dict = dict(reference_values[col_name]) # trimmed means or median according to location
    df_train[col_name].fillna(df_train['Location'].map(var_dict), inplace = True)

#%%
# Check remaining NaN
df_train[numeric_columns].isnull().sum() / len(df_train)

# %%

#----    03 Scaling data    ----#

# Get columns to scale
scaled_columns = numeric_columns.copy()

scaled_columns = scaled_columns.drop(
    ['Rainfall', 'Evaporation', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
    'RainToday01', 'RainTomorrow01',  'DiffPrev', 'DiffNext'])

scaled_columns

# %%
# Scale columns using standard scaler

scaler_data = StandardScaler()
scaled_data = scaler_data.fit_transform(df_train[scaled_columns])

scaled_data = pd.DataFrame(scaled_data, columns= [name + 'Scale' for name in scaled_columns])
scaled_data.set_index(df_train.index, inplace=True)

# print(df_train.shape)
# print(scaled_data.shape)

df_train = pd.concat([df_train, scaled_data], axis=1)

# %%

#----    04 Code Wind Direction    ----#

# Code wind direction on unit circle

wind_dir_dict = {
    'N':0,
    'NNE': 45/2,
    'NE': 45,
    'ENE': 45 + 45/2,
    'E': 90,
    'ESE': 90 + 45/2,
    'SE': 135,
    'SSE': 135 + 45/2,
    'S': 180,
    'SSW': 180 + 45/2,
    'SW': 225,
    'WSW': 225 + 45/2,
    'W': 270,
    'WNW': 270 + 45/2,
    'NW': 315,
    'NNW': 315 + 45/2,
    'Missing': np.nan
}

wind_data = pd.DataFrame(wind_dir_dict.items(), columns=['Dir', 'Degrees'])
wind_data.set_index('Dir', inplace=True)

wind_data['Rad'] = wind_data['Degrees'].map(lambda x: (x/360) * 2 * np.pi)
wind_data['Sin'] = wind_data['Rad'].map(np.sin)
wind_data['Cos'] = wind_data['Rad'].map(np.cos)

# set missing to 0 in both sin and Cos
wind_data.fillna(0, inplace=True)

# Note that reference all all rotated compared to unit circle (0 is N instead of E) 
# and also rotation is opposite
wind_data

# %%
# create Sin and Cos var to map wind direction measures

var_wind = ['WindGustDir', 'WindDir9am', 'WindDir3pm']

for col_name in var_wind:
    new_col_sin = col_name + 'Sin'
    new_col_cos = col_name + 'Cos'

    df_train[new_col_sin] = df_train[col_name].map(dict(wind_data['Sin']))
    df_train[new_col_cos] = df_train[col_name].map(dict(wind_data['Cos']))

# %%

#----    05 Test Dataset    ----#

# %%
# Missing categorical values
df_test[object_columns] = df_test[object_columns].fillna('Missing')

# %%
# Missing continuos variabels
for col_name in reference_values.columns:
    var_dict = dict(reference_values[col_name]) # trimmed means or median according to location
    df_test[col_name].fillna(df_test['Location'].map(var_dict), inplace = True)

# %%
# Scaling data according to train values
scaled_test_data = scaler_data.transform(df_test[scaled_columns])

scaled_test_data = pd.DataFrame(scaled_test_data, columns= [name + 'Scale' for name in scaled_columns])
scaled_test_data.set_index(df_test.index, inplace=True)

# print(df_test.shape)
# print(scaled_test_data.shape)

df_test = pd.concat([df_test, scaled_test_data], axis=1)

# %% Code wind direction
for col_name in var_wind:
    new_col_sin = col_name + 'Sin'
    new_col_cos = col_name + 'Cos'

    df_test[new_col_sin] = df_test[col_name].map(dict(wind_data['Sin']))
    df_test[new_col_cos] = df_test[col_name].map(dict(wind_data['Cos']))


#----    06 End Engineering    ----#

#%%
# Check values train
df_train.describe()

# %%

# Check NaN train
df_train.info()

#%%
# Check values test
df_test.describe()

# %%

# Check NaN test
df_test.info()

# %%
# Save Objects
d = shelve.open(os.path.join(os.path.dirname(sys.path[0]),'outputs', '02_feature_engineering'))
d['df_train'] = df_train
d['df_test'] = df_test
d['object_columns'] = object_columns
d['numeric_columns'] = numeric_columns
d['reference_values'] = reference_values

d['cols_smart_fillna'] = cols_smart_fillna
d['scaled_columns'] = scaled_columns
d['scaler_data'] = scaler_data
d['wind_data'] = wind_data
d['var_wind'] = var_wind
d.close()
