#=================================#
#====    03 Logistic Model    ====#
#=================================#

# 01 Encode Data
# 02 Fit Model
# 03 Grid Search
# 04 Best Model
# 05 Encode Data Advanced
# 06 Fit Model Advanced
# 07 Grid Search Advanced
# 08 Best Model Advanced
# 09 Models Comparison

#%%

#----    Settings    ----#

import sys
import shelve
from time import time
sys.path.append('../mycode')

import numpy as np
import pandas as pd
from scipy import stats 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, fbeta_score,\
    PrecisionRecallDisplay, make_scorer, precision_recall_curve, average_precision_score
from sklearn.model_selection import KFold, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

import utils  # from mycode
import myPlots  # from mycode

# get data

d_old = shelve.open("../outputs/02_feature_engineering")
df_train = d_old['df_train'].copy()
df_test = d_old['df_test'].copy()
scaler_data = d_old['scaler_data']
d_old.close()

#%%

#----    01 Encode Data    ----#

# List of all variables 
df_train.info()

# %%
# Categorical variables to include in the model
logistic_categ_var = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']

# Create encoder
logistic_var_encoder = OneHotEncoder(categories='auto', sparse = False)
logistic_var_encoder.fit(df_test[logistic_categ_var])

encoded_columns = logistic_var_encoder.get_feature_names_out(logistic_categ_var)

# %%
# Get encoded categorical data
encoded_data = logistic_var_encoder.transform(df_train[logistic_categ_var])
encoded_data = pd.DataFrame(encoded_data, columns = encoded_columns, index = df_train.index)

# %%
# Numerical variables to include in the model
logistic_numeric_var = [
    'MinTempScale', 'MaxTempScale', 
    'RainfallLogScale', 'EvaporationLogScale',
    'SunshineScale', 'WindGustSpeedLogScale', 
    'WindSpeed9amLogScale', 'WindSpeed3pmLogScale',
    'Humidity9amScale', 'Humidity3pmScale', 
    'Pressure9amScale', 'Pressure3pmScale',
    'Cloud9amScale', 'Cloud3pmScale',
    'Temp9amScale', 'Temp3pmScale',
    'YearScale', 'WeekScale']

# %%
# Check correlation variables
plt.figure(figsize=(20,20))

# sns.heatmap(df.corr(), linewidths=.5)
sns.heatmap(df_train[logistic_numeric_var].corr().abs(), annot=True, fmt = '.2f', linewidths=.5)\
    .set_title('Absolute Corretlation Value')

#%%

# Create X and y for the logistic model
logistic_X = pd.concat([df_train[logistic_numeric_var], encoded_data], axis=1)
logistic_y = df_train['RainTomorrow01']

print(logistic_X.shape)
print(logistic_y.shape)

# %%
# Encode data test

# Get encoded categorical data test
encoded_data_test = logistic_var_encoder.transform(df_test[logistic_categ_var])
encoded_data_test = pd.DataFrame(encoded_data_test, columns = encoded_columns, index = df_test.index)

# Create X_test and y_test for the logistic model
logistic_X_test = pd.concat([df_test[logistic_numeric_var], encoded_data_test], axis=1)
logistic_y_test = df_test['RainTomorrow01']

print(logistic_X_test.shape)
print(logistic_y_test.shape)

# %%

#----    02 Fit Model    ----#

# First trial logistic regression

fit_logistic = LogisticRegression(
    penalty='none', C = 1, class_weight = 'balanced', random_state = 2022) \
    .fit(logistic_X, logistic_y)

#%%
# Get info score including confsion matrix, classification report, and f2 values
utils.get_score_report(fit_logistic, logistic_y, logistic_X)

# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    fit_logistic, logistic_X, logistic_y, name="Initial Model"
)

#%%
#----    03 Grid Search    ----#

# Define f2 as scorer
ftwo_scorer = make_scorer(fbeta_score, pos_label = 1,  beta=2)

#%%
# Define grid search
param_grid = {
    'C': [.1, 1, 10], 
    'max_iter': [200],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'class_weight':[{0:1, 1:2}, 'balanced', {0:1, 1:5}],
    'random_state':[2020]
}

param_grid_II = {
    'C': [.1, 1, 10], 
    'max_iter': [200],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'class_weight':[{0:1, 1:5}, {0:1, 1:7}, {0:1, 1:9}],
    'random_state':[2020]
}

param_grid_III = {
    'C': [.01, .1, 1], 
    'max_iter': [200],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'class_weight':[{0:1, 1:5.5}, {0:1, 1:6}, {0:1, 1:6.5}],
    'random_state':[2020]
}

grid_logistic = GridSearchCV(
    LogisticRegression(), 
    param_grid=param_grid_III,
    scoring=ftwo_scorer, 
    cv = KFold(5, shuffle=True, random_state=2022)
)

#%%
%%time
grid_logistic.fit(logistic_X, logistic_y)


#%%
# Check Results
pd.DataFrame(grid_logistic.cv_results_).sort_values(by=['rank_test_score'])


#%%

#----    04 Best Model    ----#
# 'C': .1;    'class_weight': {0:1, 1:6}
best_fit_logistic = grid_logistic.best_estimator_

#%%
# Get info score including confsion matrix, classification report, and f2 values
utils.get_score_report(best_fit_logistic, logistic_y, logistic_X)

# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    best_fit_logistic, logistic_X, logistic_y, name="Best Model"
)

# %%

#----    05 Encode Data Advanced Feature   ----#

# List of all variables 
df_train.info()

# %%
# We use the same categorical variables as before
# ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']

encoded_data.shape

# %%
# Different Numerical variables to include in the model
# Include variables coded as Diff to riduce collinearity
logistic_numeric_var_adv = [
    'MaxMinDiffScale', 'MaxTempScale', 
    'RainfallLogScale', 'EvaporationLogScale',
    'SunshineScale', 'WindGustSpeedLogScale', 
    'WindSpeed9amLogScale', 'WindSpeed3pmLogScale',
    'HumidityDiffScale', 'Humidity3pmScale', 
    'PressureDiffScale', 'Pressure3pmScale',
    'Cloud9amScale', 'Cloud3pmScale',
    'Temp9amScale', 'TempDiffScale',
    'YearScale', 'WeekScale']

# %%
# Check correlation variables
plt.figure(figsize=(20,20))

# sns.heatmap(df.corr(), linewidths=.5)
sns.heatmap(df_train[logistic_numeric_var_adv].corr().abs(), annot=True, fmt = '.2f', linewidths=.5)\
    .set_title('Absolute Corretlation Value')

#%%

# Create X for the logistic model advanced
logistic_X_adv = pd.concat([df_train[logistic_numeric_var_adv], encoded_data], axis=1)

print(logistic_X_adv.shape)
print(logistic_y.shape)

# %%
# Encode data test

# Same encoded categorical data test
encoded_data_test.shape

# Create X_test and y_test for the logistic model
logistic_X_test_adv = pd.concat([df_test[logistic_numeric_var_adv], encoded_data_test], axis=1)

print(logistic_X_test_adv.shape)
print(logistic_y_test.shape)

# %%

#----    06 Fit Model Advanced    ----#

# First trial logistic regression advanced

fit_logistic_adv = LogisticRegression(
    penalty='none', C = 1, class_weight = 'balanced', random_state = 2022) \
    .fit(logistic_X_adv, logistic_y)

#%%
# Get info score including confsion matrix, classification report, and f2 values
# So advance that is worse than before 
utils.get_score_report(fit_logistic_adv, logistic_y, logistic_X_adv) 


# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    fit_logistic_adv, logistic_X_adv, logistic_y, name="Initial Model"
)

#%%
#----    07 Grid Search Advanced    ----#

# Use same f2 scorer
ftwo_scorer

#%%
# Define grid search
param_grid = {
    'C': [.1, 1, 10], 
    'max_iter': [200],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'class_weight':[{0:1, 1:2}, 'balanced', {0:1, 1:5}],
    'random_state':[2020]
}

param_grid_II = {
    'C': [.1, 1, 10], 
    'max_iter': [200],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'class_weight':[{0:1, 1:5}, {0:1, 1:7}, {0:1, 1:9}],
    'random_state':[2020]
}

param_grid_III = {
    'C': [.001, .01, .1], 
    'max_iter': [200],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'class_weight':[{0:1, 1:6}, {0:1, 1:6.5}, {0:1, 1:7}],
    'random_state':[2020]
}

grid_logistic_adv = GridSearchCV(
    LogisticRegression(), 
    param_grid=param_grid_III,
    scoring=ftwo_scorer, 
    cv = KFold(5, shuffle=True, random_state=2022)
)

#%%
%%time
grid_logistic_adv.fit(logistic_X_adv, logistic_y)


#%%
# Check Results
pd.DataFrame(grid_logistic_adv.cv_results_).sort_values(by=['rank_test_score'])


#%%

#----    08 Best Model Advanced    ----#

# 'C': .01;    'class_weight': {0:1, 1:6.5}
best_fit_logistic_adv = grid_logistic_adv.best_estimator_

#%%
# Get info score including confsion matrix, classification report, and f2 values
utils.get_score_report(best_fit_logistic_adv, logistic_y, logistic_X_adv)

# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    best_fit_logistic_adv, logistic_X_adv, logistic_y, name="Best Model"
)


#%%
#----    09 Models Comparison    ----#

# 'Simple' model on test data
utils.get_score_report(best_fit_logistic, logistic_y_test, logistic_X_test)

# %%
# 'Advanced' model on test data
utils.get_score_report(best_fit_logistic_adv, logistic_y_test, logistic_X_test_adv)

# The model 'advanced' is slightly slightly bettter

# %%
# Precision-Recall Plot

myPlots.plot_precision_recall(
    list_classifier = [best_fit_logistic, best_fit_logistic_adv],
    list_X =[logistic_X_test, logistic_X_test_adv],
    true_y = logistic_y_test,
    list_names = ['Simple Model', 'Adv Model'], 
    pos_label = 1
)

# %% 
# Check Models Coefficient 
utils.get_coef_importance(
    classifier = best_fit_logistic,
    col_names = logistic_X_test.columns
    )

# %% 
# Check Models Coefficient 
utils.get_coef_importance(
    classifier = best_fit_logistic_adv,
    col_names = logistic_X_test_adv.columns
    )

# %%

#----    10 End Logistic    ----#

# Save Objects
d = shelve.open("../outputs/03_logistic_reg")
d['logistic_X_test'] = logistic_X_test
d['logistic_X_test_adv'] = logistic_X_test_adv
d['logistic_y_test'] = logistic_y_test

d['best_fit_logistic'] = best_fit_logistic
d['best_fit_logistic_adv'] = best_fit_logistic_adv
d.close()

# %%
