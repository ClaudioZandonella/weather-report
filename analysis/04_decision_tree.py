#================================#
#====    04 Decision Tree    ====#
#================================#

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
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report, fbeta_score,\
    PrecisionRecallDisplay, make_scorer, precision_recall_curve
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
tree_categ_var = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']

# Create encoder
tree_var_encoder = OneHotEncoder(categories='auto', sparse = False)
tree_var_encoder.fit(df_test[tree_categ_var])

encoded_columns = tree_var_encoder.get_feature_names_out(tree_categ_var)

# %%
# Get encoded categorical data
encoded_data = tree_var_encoder.transform(df_train[tree_categ_var])
encoded_data = pd.DataFrame(encoded_data, columns = encoded_columns, index = df_train.index)

# %%
# Numerical variables to include in the model 
# [not scaled to facilitate interpretation]
# Year is excluded
tree_numeric_var = [
    'MinTemp', 'MaxTemp', 
    'RainfallLog', 'EvaporationLog',
    'Sunshine', 'WindGustSpeedLog', 
    'WindSpeed9amLog', 'WindSpeed3pmLog',
    'Humidity9am', 'Humidity3pm', 
    'Pressure9am', 'Pressure3pm',
    'Cloud9am', 'Cloud3pm',
    'Temp9am', 'Temp3pm',
    'Week']

# %%
# Check correlation variables
plt.figure(figsize=(20,20))

# sns.heatmap(df.corr(), linewidths=.5)
sns.heatmap(df_train[tree_numeric_var].corr().abs(), annot=True, fmt = '.2f', linewidths=.5)\
    .set_title('Absolute Corretlation Value')

#%%

# Create X and y for the decision tree
tree_X = pd.concat([df_train[tree_numeric_var], encoded_data], axis=1)
tree_y = df_train['RainTomorrow01']

print(tree_X.shape)
print(tree_y.shape)

# %%
# Encode data test

# Get encoded categorical data test
encoded_data_test = tree_var_encoder.transform(df_test[tree_categ_var])
encoded_data_test = pd.DataFrame(encoded_data_test, columns = encoded_columns, index = df_test.index)

# Create X_test and y_test for the tree model
tree_X_test = pd.concat([df_test[tree_numeric_var], encoded_data_test], axis=1)
tree_y_test = df_test['RainTomorrow01']

print(tree_X_test.shape)
print(tree_y_test.shape)

# %%

#----    02 Fit Model    ----#

# First trial decision tree

fit_tree = tree.DecisionTreeClassifier(
    criterion = 'gini', splitter = 'best', max_depth = 25,\
    class_weight = 'balanced', random_state = 2022) \
    .fit(tree_X, tree_y)

#%%
# Get info score including confsion matrix, classification report, and f2 values
utils.get_score_report(fit_tree, tree_y, tree_X)

# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    fit_tree, tree_X, tree_y, name="Initial Model"
)

# %%
# Plot tree
plt.figure(figsize=(40, 20))
annotations = tree.plot_tree(fit_tree, feature_names = tree_X.columns, filled = True, max_depth = 3)
#%%
#----    03 Grid Search    ----#

# Define f2 as scorer
ftwo_scorer = make_scorer(fbeta_score, pos_label = 1,  beta=2)

#%%
# Define grid search
param_grid = {
    'criterion' : ['gini'], 
    'splitter' : ['best'], 
    'max_depth' : [5, 10, 15, 20, 25],
    'class_weight' : ['balanced'], 
    'random_state' : [2022]
}

param_grid_II = {
    'criterion' : ['gini'], 
    'splitter' : ['best'], 
    'max_depth' : [5, 8, 10, 12, 15],
    'class_weight' : ['balanced', {0:1, 1:5},  {0:1, 1:7}], 
    'random_state' : [2022]
}

param_grid_III = {
    'criterion' : ['gini'], 
    'splitter' : ['best'], 
    'max_depth' : [7, 8, 9, 10, 11],
    'class_weight' : [{0:1, 1:5}, {0:1, 1:6}, {0:1, 1:7}, {0:1, 1:8}], 
    'random_state' : [2022]
}

grid_tree = GridSearchCV(
    tree.DecisionTreeClassifier(), 
    param_grid=param_grid_III,
    scoring=ftwo_scorer, 
    cv = KFold(5, shuffle=True, random_state=2022)
)

#%%
%%time
grid_tree.fit(tree_X, tree_y)


#%%
# Check Results
grid_tree_result = pd.DataFrame(grid_tree.cv_results_).sort_values(by=['rank_test_score'])
myPlots.plot_grid_tree(grid_tree_result)
grid_tree_result

#%%

#----    04 Best Model    ----#

# 'max_depth': 8;    'class_weight': {0:1, 1:6}
best_fit_tree = grid_tree.best_estimator_

#%%
# Get info score including confsion matrix, classification report, and f2 values
utils.get_score_report(best_fit_tree, tree_y, tree_X)

# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    best_fit_tree, tree_X, tree_y, name="Best Model"
)

# %%
# Plot tree and feature importance
plt.figure(figsize=(40, 20))
annotations = tree.plot_tree(best_fit_tree, feature_names = tree_X.columns, filled = True, max_depth = 3)

utils.get_feature_importance(best_fit_tree, tree_X.columns)

# %%

#----    05 Encode Data Advanced Feature   ----#

# List of all variables 
df_train.info()

# %%
# Categorical variables to include in the model
tree_categ_var_adv = ['Location', 'RainToday']

# Create encoder
tree_var_encoder_adv = OneHotEncoder(categories='auto', sparse = False)
tree_var_encoder_adv.fit(df_test[tree_categ_var_adv])

encoded_columns_adv = tree_var_encoder_adv.get_feature_names_out(tree_categ_var_adv)

# %%
# Get encoded categorical data
encoded_data_adv = tree_var_encoder_adv.transform(df_train[tree_categ_var_adv])
encoded_data_adv = pd.DataFrame(encoded_data_adv, columns = encoded_columns_adv, index = df_train.index)
encoded_data_adv.shape

# %%
# Different Numerical variables to include in the model
# Include variables coded as Diff to riduce collinearity
# Include wind directions as Cos and Sin
tree_numeric_var_adv = [
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

# %%
# Check correlation variables
plt.figure(figsize=(20,20))

# sns.heatmap(df.corr(), linewidths=.5)
sns.heatmap(df_train[tree_numeric_var_adv].corr().abs(), annot=True, fmt = '.2f', linewidths=.5)\
    .set_title('Absolute Corretlation Value')

#%%

# Create X for the tree model advanced
tree_X_adv = pd.concat([df_train[tree_numeric_var_adv], encoded_data_adv], axis=1)

print(tree_X_adv.shape)
print(tree_y.shape)

# %%
# Encode data test

# Get encoded categorical data test
encoded_data_test_adv = tree_var_encoder_adv.transform(df_test[tree_categ_var_adv])
encoded_data_test_adv = pd.DataFrame(encoded_data_test_adv, columns = encoded_columns_adv, index = df_test.index)

# Create X_test and y_test for the tree model
tree_X_test_adv = pd.concat([df_test[tree_numeric_var_adv], encoded_data_test_adv], axis=1)

print(tree_X_test_adv.shape)
print(tree_y_test.shape)

#%%

#----    06 Fit Model Advanced    ----#

# First trial decision tree

fit_tree_adv = tree.DecisionTreeClassifier(
    criterion = 'gini', splitter = 'best', max_depth = 25,\
    class_weight = 'balanced', random_state = 2022) \
    .fit(tree_X_adv, tree_y)

#%%
# Get info score including confsion matrix, classification report, and f2 values
utils.get_score_report(fit_tree_adv, tree_y, tree_X_adv)

# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    fit_tree_adv, tree_X_adv, tree_y, name="Initial Model"
)

# %%
# Plot tree
plt.figure(figsize=(40, 20))
annotations = tree.plot_tree(fit_tree_adv, feature_names = tree_X_adv.columns, filled = True, max_depth = 3)

#%%
#----    07 Grid Search Advanced    ----#

# Again we use f2 as scorer
ftwo_scorer

#%%
# Define grid search
param_grid = {
    'criterion' : ['gini'], 
    'splitter' : ['best'], 
    'max_depth' : [5, 10, 15, 20, 25],
    'class_weight' : ['balanced'], 
    'random_state' : [2022]
}

param_grid_II = {
    'criterion' : ['gini'], 
    'splitter' : ['best'], 
    'max_depth' : [5, 8, 10, 12, 15],
    'class_weight' : ['balanced', {0:1, 1:5},  {0:1, 1:7}], 
    'random_state' : [2022]
}

param_grid_III = {
    'criterion' : ['gini'], 
    'splitter' : ['best'], 
    'max_depth' : [7, 8, 9, 10, 11],
    'class_weight' : [{0:1, 1:5}, {0:1, 1:6}, {0:1, 1:7}, {0:1, 1:8}], 
    'random_state' : [2022]
}

grid_tree_adv = GridSearchCV(
    tree.DecisionTreeClassifier(), 
    param_grid=param_grid_III,
    scoring=ftwo_scorer, 
    cv = KFold(5, shuffle=True, random_state=2022)
)

#%%
%%time
grid_tree_adv.fit(tree_X_adv, tree_y)


#%%
# Check Results
grid_tree_result_adv = pd.DataFrame(grid_tree_adv.cv_results_).sort_values(by=['rank_test_score'])
myPlots.plot_grid_tree(grid_tree_result_adv)
grid_tree_result_adv

#%%

#----    08 Best Model Advanced    ----#

# 'max_depth': 8;    'class_weight': {0:1, 1:6}
best_fit_tree_adv = grid_tree_adv.best_estimator_

#%%
# Get info score including confsion matrix, classification report, and f2 values
utils.get_score_report(best_fit_tree_adv, tree_y, tree_X_adv)

# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    best_fit_tree_adv, tree_X_adv, tree_y, name="Best Model"
)

# %%
# Plot tree and feature importance
plt.figure(figsize=(40, 20))
annotations = tree.plot_tree(best_fit_tree_adv, feature_names = tree_X_adv.columns, filled = True, max_depth = 3)

utils.get_feature_importance(best_fit_tree_adv, tree_X_adv.columns)

#%%
#----    09 Models Comparison    ----#

# 'Simple' model on test data
utils.get_score_report(best_fit_tree, tree_y_test, tree_X_test)

# %%
# 'Advanced' model on test data
utils.get_score_report(best_fit_tree_adv, tree_y_test, tree_X_test_adv)

# The model 'advanced' is slightly slightly bettter

# %%
# Precision-Recall Plot

myPlots.plot_precision_recall(
    list_classifier = [best_fit_tree, best_fit_tree_adv],
    list_X =[tree_X_test, tree_X_test_adv],
    true_y = tree_y_test,
    list_names = ['Simple Model', 'Adv Model'], 
    pos_label = 1
)

# %% 
# Check Models feature importance 
utils.get_feature_importance(
    classifier = best_fit_tree,
    col_names = tree_X_test.columns
    )

# %% 
# Check Models feature importance 
utils.get_feature_importance(
    classifier = best_fit_tree_adv,
    col_names = tree_X_test_adv.columns
    )

# %%

#----    10 End Tree    ----#

# Save Objects
d = shelve.open("../outputs/04_decision_tree")

# Training Data
d['tree_X'] = tree_X
d['tree_X_adv'] = tree_X_adv
d['tree_y'] = tree_y

# Test Data
d['tree_X_test'] = tree_X_test
d['tree_X_test_adv'] = tree_X_test_adv
d['tree_y_test'] = tree_y_test

# Models
d['best_fit_tree'] = best_fit_tree
d['best_fit_tree_adv'] = best_fit_tree_adv
d.close()