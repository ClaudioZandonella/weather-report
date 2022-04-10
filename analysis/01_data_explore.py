#===============================#
#====    01 Data Explore    ====#
#===============================#

# Index Content
# 01 Import data
# 02 Descriptive Stat Categorical
# 03 Descriptive Stat Continuos
# 04 Descriptive Dates
# 05 End Descriptive

# %%

#----    Settings    ----#

import sys, os
import shelve
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'mycode'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from mycode/
import utils


# %%

#----    01 Import data    ----#

df = pd.read_csv(os.path.join(os.path.dirname(sys.path[0]),'data', 'weather.csv'))
df.drop(['Unnamed: 0'], axis=1, inplace = True)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(['Location', 'Date'], inplace = True)
df.reset_index(drop = True, inplace = True)

# %%
# Data type info
df.info()

# %%

#----    02 Descriptive Stat Categorical    ----#

# Check categorial variables number of levels
df.describe(include=object)

# %%
# Check missing values proportion within to location
object_columns = df.select_dtypes(include=['object']).columns[1:]
pd.concat([df.Location, df[object_columns].isnull()], axis=1)\
    .groupby('Location').mean()

# %%
# Location
utils.my_count(df['Location'])

# %%
# WindGustDir
utils.my_count(df['WindGustDir'])

# %%
# WindDir9am
utils.my_count(df['WindDir9am'])

# %%
# WindDir3pm
utils.my_count(df['WindDir3pm'])

# %%
# RainToday
utils.my_count(df['RainToday'])

# %%
# RainTomorrow
utils.my_count(df['RainTomorrow'])

# %%

# # Check if missing values for RainToday may be coded in RainTomorrow of yesterday
# df_null_rain = df[df['RainToday'].isnull()]
# df_null_rain['Yesterday'] = df_null_rain['Date'] - np.timedelta64(1, 'D')

# len_null_rain = len(df_null_rain)

# test = [-1]*len_null_rain


# for i in np.arange(len_null_rain):
#     yesterday, location = df_null_rain.iloc[i][['Yesterday', 'Location']]
#     value = len(df[(df['Location'] == location) & (df['Date'] == yesterday)])
#     test[i] = value

# np.mean(test) # no the data is clean

# %%

#----    03 Descriptive Stat Continuos    ----#

# Summary continuos variables
df.describe()

# %%
# NaN proportion
numeric_columns = df.select_dtypes(include=np.number).columns
df[numeric_columns].isnull().sum() / len(df)

# %%
# Check missing values proportion within to location
pd.concat([df.Location, df[numeric_columns].isnull()], axis=1)\
    .groupby('Location').mean()
# %%
# Plot distributions

fig, axes = plt.subplots(8,2, figsize= (14,30))

for i, col_name in enumerate(numeric_columns):
    # df.pivot(columns="RainTomorrow", values=col_name).plot.hist(ax=axes[i%8][i//8], alpha=0.6, bins = 30);
    df[col_name].hist(ax=axes[i%8][i//8], alpha=0.7, bins = 30);
    axes[i%8][i//8].set_title(f'{col_name}', fontsize=13);
    plt.subplots_adjust(hspace=0.45)

# %%
# Check tails of Rainfall; Evaporation; WindGustSpeed; WindSpeed9am; WindSpeed3pm 

dic_filter = {'Rainfall':20, 
              'Evaporation':15, 
              'WindGustSpeed':80, 
              'WindSpeed9am':40, 
              'WindSpeed3pm':40}

fig, axes = plt.subplots(3,2, figsize= (14,14))

for i, col_name in enumerate(dic_filter):
    df[df[col_name] >= dic_filter[col_name]][col_name].hist(ax=axes[i%3][i//3], alpha=0.7, bins = 30);
    axes[i%3][i//3].set_title(f'{col_name}', fontsize=13);
    plt.subplots_adjust(hspace=0.45)

# %%
# Get log transform Rainfall; Evaporation; WindGustSpeed; WindSpeed9am; WindSpeed3pm
for var_name in dic_filter:
    new_name = var_name + 'Log'
    df[new_name] = np.log(df[var_name] + 1)
#%%
# check log transformed values 
fig, axes = plt.subplots(3,2, figsize= (14,14))

for i, col_name in enumerate(dic_filter):
    var_name = col_name + 'Log'
    df[var_name].hist(ax = axes[i%3][i//3], alpha = 0.7, bins = 30);
    axes[i%3][i//3].set_title(f'{var_name}', fontsize=13);
    plt.subplots_adjust(hspace = 0.45)

#%%
# Rainfall check values different from 0
df['RainfallLog'].where(lambda x: x != 0).hist(alpha=0.7, bins = 30)

#%%
# Relation between variabels
plt.figure(figsize=(20,20))

# sns.heatmap(df.corr(), linewidths=.5)
sns.heatmap(df.corr().abs(), annot=True, fmt = '.2f', linewidths=.5)\
    .set_title('Absolute Corretlation Value')

# %%
# Check high values of correlation
df_corr = df.corr().round(2)
df_corr[abs(df_corr) > .6]

#%%
# Check if both values are available for matching variables

# Temperature
print(df[['MinTemp', 'MaxTemp']].isnull().sum(axis=1).value_counts())
print(df[['Temp9am', 'Temp3pm']].isnull().sum(axis=1).value_counts())

df['TempDiff'] = df['Temp3pm'] - df['Temp9am']
df['MaxMinDiff'] = df['MaxTemp'] - df['MinTemp']
df['TempDiff'].hist()
df['MaxMinDiff'].hist()

df[['Temp9am', 'Temp3pm', 'MinTemp', 'MaxTemp', 'TempDiff', 'MaxMinDiff']].corr().round(2)

#%%

# Pressure
df[['Pressure3pm', 'Pressure9am']].isnull().sum(axis=1).value_counts()
df['PressureDiff'] = df['Pressure3pm'] - df['Pressure9am']
df['PressureDiff'].hist()
df[['Pressure3pm', 'Pressure9am', 'PressureDiff']].corr().round(2)

#%%

# Humidity
df[['Humidity3pm', 'Humidity9am']].isnull().sum(axis=1).value_counts()
df['HumidityDiff'] = df['Humidity3pm'] - df['Humidity9am']
df['HumidityDiff'].hist()
df[['Humidity3pm', 'Humidity9am', 'HumidityDiff']].corr().round(2)

# %%
# Code rain 0-1
df['RainToday01'] = df['RainToday'].map({'No' : 0, 'Yes' : 1})
df['RainTomorrow01'] = df['RainTomorrow'].map({'No' : 0, 'Yes' : 1})

# print(pd.value_counts(df['RainToday']))
# print(pd.value_counts(df['RainToday_01']))

# print(pd.value_counts(df['RainTomorrow']))
# print(pd.value_counts(df['RainTomorrow_01']))

# %%
# Relation between variabels
plt.figure(figsize = (20,20))
# sns.heatmap(df.corr(), linewidths=.5)
sns.heatmap(df.corr().abs(), linewidths = .5, annot = True, fmt = '.2f')\
    .set_title('Absolute Corretlation Value')

# Check high values of correlation
df_corr = df.corr().round(2)
df_corr[abs(df_corr) > .6]

# %%

#----    04 Descriptive Dates    ----#

#%%
# Get Date values separately
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week

# %%

# Check availabe years
print(utils.my_count(df['Year']))

# Check dates are continuos
grouped_location = df.groupby(['Location'])
grouped_location['Year'].value_counts().unstack()

# %%
# Check diff in dates

df['DiffPrev'] = utils.get_days_diff(df, periods = 1)
print(grouped_location['DiffPrev'].value_counts().unstack())

df['DiffNext'] = utils.get_days_diff(df, periods = -1)
print(grouped_location['DiffNext'].value_counts().unstack())

# %%
# no duplicate dates within Locations
all(grouped_location['Date'].apply(lambda x: len(x)/len(x.unique()))==1)


# %%
# Check strange value Melbourne
df.loc[(df['Location'] == "Melbourne") & (df['DiffPrev']> 100)]
df.loc[(df['Location'] == "Melbourne") & (df['Date'] > '2015-01-01') & (df['Date'] < '2016-05-05')]

# %%

#----    05 End Descriptive    ----#

df.info()

# %%
df.describe()

# %%
# Save Objects
d = shelve.open(os.path.join(os.path.dirname(sys.path[0]),'outputs', '01_data_explore'))
d['df'] = df
d.close()


# %%
