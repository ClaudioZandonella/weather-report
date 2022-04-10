#==========================#
#====    05 XGBoost    ====#
#==========================#

# 01 Get Data
# 02 Encode Data
# 03 Fit Model
# 04 Grid Search
# 05 Best Model
# 06 Encode Data Advanced
# 07 Fit Model Advanced
# 08 Grid Search Advanced
# 09 Best Model Advanced
# 10 Models Comparison

#%%

#----    Settings    ----#

import sys
import shelve
from time import time
sys.path.append('../mycode')

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import fbeta_score, make_scorer, \
    PrecisionRecallDisplay
from sklearn.model_selection import KFold, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# from mycode
import utils  
import myStats
import myPlots

# get data

# Initial data
d_old = shelve.open("../outputs/01_data_explore")
df = d_old['df'].copy()
d_old.close()

# Feature engineering
d_old = shelve.open("../outputs/02_feature_engineering")
wind_data = d_old['wind_data']
var_wind = d_old['var_wind']
d_old.close()


#%%
#----    01 Get Data    ----#

# XGBoost handle missing data so we use original data with missing values

# %% 
# Code wind direction
for col_name in var_wind:
    new_col_sin = col_name + 'Sin'
    new_col_cos = col_name + 'Cos'

    df[new_col_sin] = df[col_name].map(dict(wind_data['Sin']))
    df[new_col_cos] = df[col_name].map(dict(wind_data['Cos']))


# %%
# Split train and train data
df_train = df[df['Year'] <= 2015].copy()
df_test = df[df['Year'] > 2015].copy()

#%%
print(df_train.shape)
print(df_test.shape)

#%%
#----    02 Encode Data    ----#

# Use same variables used in the decision tree analysis

# %%
# Categorical variables to include in the model
xgb_categ_var = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']

# %%
# Numerical variables to include in the model 
# [not scaled to facilitate interpretation]
# Year is excluded
xgb_numeric_var = [
    'MinTemp', 'MaxTemp', 
    'RainfallLog', 'EvaporationLog',
    'Sunshine', 'WindGustSpeedLog', 
    'WindSpeed9amLog', 'WindSpeed3pmLog',
    'Humidity9am', 'Humidity3pm', 
    'Pressure9am', 'Pressure3pm',
    'Cloud9am', 'Cloud3pm',
    'Temp9am', 'Temp3pm',
    'Week']

#%%
# encode data
xgb_X, xgb_X_test = utils.get_encoded_data(
    df_train = df_train,
    df_test = df_test,
    categ_var = xgb_categ_var,
    numeric_var = xgb_numeric_var)

#%%
# get output
xgb_y = df_train['RainTomorrow01'].copy()
xgb_y_test = df_test['RainTomorrow01'].copy()

# %%
# Check train data
print(xgb_X.shape)
print(xgb_y.shape)

# %%
# Check test data
print(xgb_X_test.shape)
print(xgb_y_test.shape)

# %%
# Set week from int to float as required by xgboost
xgb_X['Week'] = xgb_X['Week'].astype(float)
xgb_X_test['Week'] = xgb_X_test['Week'].astype(float)

# %%

#----    03 Fit Model    ----#

# First trial xgboost

fit_xgb = xgb.XGBClassifier(
    n_estimators = 150,
    objective = 'binary:logistic',
    missing = np.nan,
    scale_pos_weight = 3.5,
    max_depth = 6,
    colsample_bylevel = 0.5,
    use_label_encoder = False,
    seed = 2022
    ).fit(xgb_X, xgb_y)

#%%
# Get info score including confsion matrix, classification report, and f2 values
myStats.get_score_report(fit_xgb, xgb_y, xgb_X)

# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    fit_xgb, xgb_X, xgb_y, name = "Initial Model"
)

#%%
#----    04 Grid Search    ----#

# Define f2 as scorer
ftwo_scorer = make_scorer(fbeta_score, pos_label = 1,  beta = 2)

#%%
# Define grid search
param_grid = {
    'n_estimators': [80, 100, 120, 150],
    'max_depth' : [6], 
    'learning_rate' : [.3], 
    'gamma' : [0],
    'reg_lambda': [1],
    'scale_pos_weight' : [6]
}

param_grid_II = {
    'max_depth' : [4, 6, 8], 
    'learning_rate' : [.1, .3, .5], 
    'gamma' : [0],
    'reg_lambda': [1],
    'scale_pos_weight' : [6]
}

param_grid_III = {
    'max_depth' : [8, 10, 12], 
    'learning_rate' : [.01, .05, .1], 
    'gamma' : [0],
    'reg_lambda': [1],
    'scale_pos_weight' : [6]
}

param_grid_IV = {
    'max_depth' : [7, 8, 9], 
    'learning_rate' : [.075, .1, .2], 
    'gamma' : [0],
    'reg_lambda': [1],
    'scale_pos_weight' : [6]
}

param_grid_V = {
    'max_depth' : [7], 
    'learning_rate' : [.1], 
    'gamma' : [0, .25, 1],
    'reg_lambda': [0, 1, 10],
    'scale_pos_weight' : [6]
}

param_grid_VI = {
    'max_depth' : [7], 
    'learning_rate' : [.1], 
    'gamma' : [0],
    'reg_lambda': [10, 20, 100],
    'scale_pos_weight' : [5, 6, 7]
}

param_grid_VII = {          # to allow fast compilation
    'max_depth' : [7], 
    'learning_rate' : [.1], 
    'gamma' : [0],
    'reg_lambda': [10],
    'scale_pos_weight' : [7]
}

grid_xgb = GridSearchCV(
    xgb.XGBClassifier(
        n_estimators = 120,
        objective = 'binary:logistic',
        eval_metric = 'aucpr',
        missing = np.nan,
        colsample_bytree = 0.5,
        use_label_encoder = False,
        seed = 2022
    ), 
    param_grid = param_grid_VII,
    scoring = ftwo_scorer, 
    cv = KFold(5, shuffle = True, random_state = 2022)
)

#%%
## %%time
grid_xgb.fit(xgb_X, xgb_y)


#%%
# Check Results
grid_xgb_result = pd.DataFrame(grid_xgb.cv_results_).sort_values(by = ['rank_test_score'])

sns.pointplot(data = grid_xgb_result, y = 'mean_test_score', hue = 'param_scale_pos_weight',
              x = 'param_reg_lambda')

grid_xgb_result

#%%

#----    05 Best Model    ----#

# 'max_depth' : 7; 'learning_rate' : .1; 'gamma' : 0; 'reg_lambda': 10; 'scale_pos_weight' : 7
best_fit_xgb = grid_xgb.best_estimator_

#%%
# Get info score including confsion matrix, classification report, and f2 values
myStats.get_score_report(best_fit_xgb, xgb_y, xgb_X)

# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    best_fit_xgb, xgb_X, xgb_y, name = "Best Model"
)

#%%
# Check Features
best_features = myStats.get_feature_importance(best_fit_xgb, xgb_X.columns)

# Plot Features
sns.barplot(data = best_features.head(10), x = 'importance', y = 'feature')

best_features
# %%


#----    06 Encode Data Advanced Feature   ----#

# Use same variables used in the decision tree analysis

# %%
#
# Categorical variables to include in the model
xgb_categ_var_adv = ['Location', 'RainToday']

# %%
# Different Numerical variables to include in the model
# Include variables coded as Diff to reduce collinearity
# Include wind directions as Cos and Sin
xgb_numeric_var_adv = [
    'MaxMinDiff', 'MaxTemp', 
    'RainfallLog', 'EvaporationLog',
    'Sunshine', 'WindGustSpeedLog', 
    'WindSpeed9amLog', 'WindSpeed3pmLog',
    'HumidityDiff', 'Humidity3pm', 
    'PressureDiff', 'Pressure3pm',
    'Cloud9am', 'Cloud3pm',
    'Temp9am', 'TempDiff',
    'WindGustDirSin', 'WindGustDirCos',
    'WindDir9amSin', 'WindDir9amCos',
    'WindDir3pmSin', 'WindDir3pmCos',
    'Week']

#%%
# encode data
xgb_X_adv, xgb_X_test_adv = utils.get_encoded_data(
    df_train = df_train,
    df_test = df_test,
    categ_var = xgb_categ_var_adv,
    numeric_var = xgb_numeric_var_adv)

# %%
# Check train data
print(xgb_X_adv.shape)
print(xgb_y.shape)

# %%
# Check test data
print(xgb_X_test_adv.shape)
print(xgb_y_test.shape)

#%%
# Set week from int to float as required by xgboost
xgb_X_adv['Week'] = xgb_X_adv['Week'].astype(float)
xgb_X_test_adv['Week'] = xgb_X_test_adv['Week'].astype(float)
#%%

#----    07 Fit Model Advanced    ----#

# First trial xgboost

fit_xgb_adv = xgb.XGBClassifier(
    n_estimators = 150,
    objective = 'binary:logistic',
    missing = np.nan,
    scale_pos_weight = 3.5,
    max_depth = 6,
    colsample_bylevel = 0.5,
    use_label_encoder = False,
    seed = 2022
    ).fit(xgb_X_adv, xgb_y)

#%%
# Get info score including confsion matrix, classification report, and f2 values
myStats.get_score_report(fit_xgb_adv, xgb_y, xgb_X_adv)

# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    fit_xgb_adv, xgb_X_adv, xgb_y, name = "Initial Model"
)

#%%
#----    08 Grid Search Advanced    ----#

# Again we use f2 as scorer
ftwo_scorer

#%%
# Define grid search
param_grid = {
    'n_estimators': [80, 100, 120, 150],
    'max_depth' : [6], 
    'learning_rate' : [.3], 
    'gamma' : [0],
    'reg_lambda': [1],
    'scale_pos_weight' : [6]
}

param_grid_II = {
    'max_depth' : [4, 6, 8], 
    'learning_rate' : [.5, .1, .2], 
    'gamma' : [0],
    'reg_lambda': [1],
    'scale_pos_weight' : [6]
}

param_grid_III = {
    'max_depth' : [6, 7, 8, 9], 
    'learning_rate' : [.075, .1, .3], 
    'gamma' : [0],
    'reg_lambda': [1],
    'scale_pos_weight' : [6]
}

param_grid_IV = {
    'max_depth' : [7], 
    'learning_rate' : [.1], 
    'gamma' : [0, .25, 1],
    'reg_lambda': [1, 10, 20],
    'scale_pos_weight' : [6]
}

param_grid_V = {
    'max_depth' : [7], 
    'learning_rate' : [.1], 
    'gamma' : [0],
    'reg_lambda': [10],
    'scale_pos_weight' : [5, 6, 7, 8]
}

param_grid_VI = {          # to allow fast compilation
    'max_depth' : [7], 
    'learning_rate' : [.1], 
    'gamma' : [0],
    'reg_lambda': [10],
    'scale_pos_weight' : [7]
}


grid_xgb_adv = GridSearchCV(
    xgb.XGBClassifier(
        n_estimators = 120,
        objective = 'binary:logistic',
        eval_metric = 'aucpr',
        missing = np.nan,
        colsample_bytree = 0.5,
        use_label_encoder = False,
        seed = 2022
    ), 
    param_grid = param_grid_VI,
    scoring = ftwo_scorer, 
    cv = KFold(5, shuffle = True, random_state = 2022)
)

#%%
## %%time
grid_xgb_adv.fit(xgb_X_adv, xgb_y)


#%%
# Check Results
grid_xgb_result_adv = pd.DataFrame(grid_xgb_adv.cv_results_).sort_values(by = ['rank_test_score'])

sns.pointplot(data = grid_xgb_result_adv, y = 'mean_test_score', x = 'param_scale_pos_weight',
              hue = 'param_gamma')

grid_xgb_result_adv

#%%

#----    09 Best Model Advanced    ----#

# 'max_depth' : 7; 'learning_rate' : .1; 'gamma' : 0; 'reg_lambda': 10; 'scale_pos_weight' : 7
best_fit_xgb_adv = grid_xgb_adv.best_estimator_

#%%
# Get info score including confsion matrix, classification report, and f2 values
myStats.get_score_report(best_fit_xgb_adv, xgb_y, xgb_X_adv)

# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    best_fit_xgb_adv, xgb_X_adv, xgb_y, name = "Best Model"
)


#%%
# Check Features
best_features_adv = myStats.get_feature_importance(best_fit_xgb_adv, xgb_X_adv.columns)

# Plot Features
sns.barplot(data = best_features_adv.head(10), x = 'importance', y = 'feature')

best_features_adv

#%%
#----    10 Models Comparison    ----#

# 'Simple' model on test data
myStats.get_score_report(best_fit_xgb, xgb_y_test, xgb_X_test)

# %%
# 'Advanced' model on test data
myStats.get_score_report(best_fit_xgb_adv, xgb_y_test, xgb_X_test_adv)

# The model 'advanced' is slightly slightly better

# %%
# Precision-Recall Plot

myPlots.plot_precision_recall(
    list_classifier = [best_fit_xgb, best_fit_xgb_adv],
    list_X = [xgb_X_test, xgb_X_test_adv],
    true_y = xgb_y_test,
    list_names = ['Simple Model', 'Adv Model'], 
    pos_label = 1
)

# %% 
# Check Models feature importance 
myStats.get_feature_importance(
    classifier = best_fit_xgb,
    col_names = xgb_X_test.columns
    )

# %% 
# Check Models feature importance 
myStats.get_feature_importance(
    classifier = best_fit_xgb_adv,
    col_names = xgb_X_test_adv.columns
    )

# %%

#----    11 End XGBoost    ----#

# Save Objects
d = shelve.open("../outputs/06_xgboost")

# Training Data
d['xgb_X'] = xgb_X
d['xgb_X_adv'] = xgb_X_adv
d['xgb_y'] = xgb_y

# Test Data
d['xgb_X_test'] = xgb_X_test
d['xgb_X_test_adv'] = xgb_X_test_adv
d['xgb_y_test'] = xgb_y_test

# Models
d['best_fit_xgb'] = best_fit_xgb
d['best_fit_xgb_adv'] = best_fit_xgb_adv
d.close()
#======================
# %%
