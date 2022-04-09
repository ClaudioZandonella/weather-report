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
from sklearn import tree
from sklearn.metrics import fbeta_score, make_scorer,\
     PrecisionRecallDisplay
from sklearn.model_selection import KFold, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# from mycode/
import utils  
import myStats
import myPlots

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
sns.heatmap(df_train[tree_numeric_var].corr().abs(), annot = True, fmt = '.2f', linewidths = .5)\
    .set_title('Absolute Corretlation Value')

#%%
# encode data
tree_X, tree_X_test = utils.get_encoded_data(
    df_train = df_train,
    df_test = df_test,
    categ_var = tree_categ_var,
    numeric_var = tree_numeric_var)

#%%
# get output
tree_y = df_train['RainTomorrow01'].copy()
tree_y_test = df_test['RainTomorrow01'].copy()

# %%
# Check train data
print(tree_X.shape)
print(tree_y.shape)

# %%
# Check test data
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
myStats.get_score_report(fit_tree, tree_y, tree_X)

# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    fit_tree, tree_X, tree_y, name="Initial Model"
)

# %%
# Plot tree
plt.figure(figsize = (40, 20))
annotations = tree.plot_tree(fit_tree, feature_names = tree_X.columns, filled = True, max_depth = 3)
#%%
#----    03 Grid Search    ----#

# Define f2 as scorer
ftwo_scorer = make_scorer(fbeta_score, pos_label = 1,  beta=2)

#%%
# Define grid search
param_grid = {
    'max_depth' : [5, 10, 15, 20, 25],
    'class_weight' : ['balanced']
}

param_grid_II = {
    'max_depth' : [5, 8, 10, 12, 15],
    'class_weight' : ['balanced', {0:1, 1:5},  {0:1, 1:7}]
}

param_grid_III = {
    'max_depth' : [7, 8, 9, 10, 11],
    'class_weight' : [{0:1, 1:5}, {0:1, 1:6}, {0:1, 1:7}, {0:1, 1:8}]
}

grid_tree = GridSearchCV(
    tree.DecisionTreeClassifier(
        criterion = 'gini', 
        splitter = 'best',
        random_state = 2022
    ), 
    param_grid = param_grid_III,
    scoring = ftwo_scorer, 
    cv = KFold(5, shuffle = True, random_state = 2022)
)

#%%
%%time
grid_tree.fit(tree_X, tree_y)


#%%
# Check Results
grid_tree_result = pd.DataFrame(grid_tree.cv_results_).sort_values(by = ['rank_test_score'])
myPlots.plot_grid_tree(grid_tree_result)
grid_tree_result

#%%

#----    04 Best Model    ----#

# 'max_depth': 8;    'class_weight': {0:1, 1:6}
best_fit_tree = grid_tree.best_estimator_

#%%
# Get info score including confsion matrix, classification report, and f2 values
myStats.get_score_report(best_fit_tree, tree_y, tree_X)

# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    best_fit_tree, tree_X, tree_y, name = "Best Model"
)

# %%
# Plot tree 
plt.figure(figsize = (40, 20))
annotations = tree.plot_tree(best_fit_tree, feature_names = tree_X.columns, filled = True, max_depth = 3)

#%%
# Check Features
best_features = myStats.get_feature_importance(best_fit_tree, tree_X.columns)

# Plot Features
sns.barplot(data = best_features.head(10), x = 'importance', y = 'feature')

best_features
# %%

#----    05 Encode Data Advanced Feature   ----#

# List of all variables 
df_train.info()

# %%
# Categorical variables to include in the model
tree_categ_var_adv = ['Location', 'RainToday']

# %%
# Different Numerical variables to include in the model
# Include variables coded as Diff to reduce collinearity
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
sns.heatmap(df_train[tree_numeric_var_adv].corr().abs(), annot = True, fmt = '.2f', linewidths = .5)\
    .set_title('Absolute Corretlation Value')

#%%
# encode data
tree_X_adv, tree_X_test_adv = utils.get_encoded_data(
    df_train = df_train,
    df_test = df_test,
    categ_var = tree_categ_var_adv,
    numeric_var = tree_numeric_var_adv)

# %%
# Check train data
print(tree_X_adv.shape)
print(tree_y.shape)

# %%
# Check test data
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
myStats.get_score_report(fit_tree_adv, tree_y, tree_X_adv)

# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    fit_tree_adv, tree_X_adv, tree_y, name = "Initial Model"
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
    'max_depth' : [5, 10, 15, 20, 25],
    'class_weight' : ['balanced']
}

param_grid_II = {
    'max_depth' : [5, 8, 10, 12, 15],
    'class_weight' : ['balanced', {0:1, 1:5},  {0:1, 1:7}]
}

param_grid_III = {
    'max_depth' : [7, 8, 9, 10, 11],
    'class_weight' : [{0:1, 1:5}, {0:1, 1:6}, {0:1, 1:7}, {0:1, 1:8}]
}

grid_tree_adv = GridSearchCV(
    tree.DecisionTreeClassifier(
        criterion = 'gini', 
        splitter = 'best',
        random_state = 2022
    ), 
    param_grid = param_grid_III,
    scoring = ftwo_scorer, 
    cv = KFold(5, shuffle = True, random_state = 2022)
)

#%%
%%time
grid_tree_adv.fit(tree_X_adv, tree_y)


#%%
# Check Results
grid_tree_result_adv = pd.DataFrame(grid_tree_adv.cv_results_).sort_values(by = ['rank_test_score'])
myPlots.plot_grid_tree(grid_tree_result_adv)
grid_tree_result_adv

#%%

#----    08 Best Model Advanced    ----#

# 'max_depth': 8;    'class_weight': {0:1, 1:6}
best_fit_tree_adv = grid_tree_adv.best_estimator_

#%%
# Get info score including confsion matrix, classification report, and f2 values
myStats.get_score_report(best_fit_tree_adv, tree_y, tree_X_adv)

# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    best_fit_tree_adv, tree_X_adv, tree_y, name="Best Model"
)

# %%
# Plot tree
plt.figure(figsize = (40, 20))
annotations = tree.plot_tree(best_fit_tree_adv, feature_names = tree_X_adv.columns, filled = True, max_depth = 3)

#%%
# Check Features
best_features_adv = myStats.get_feature_importance(best_fit_tree_adv, tree_X_adv.columns)

# Plot Features
sns.barplot(data = best_features_adv.head(10), x = 'importance', y = 'feature')

best_features_adv

#%%
#----    09 Models Comparison    ----#

# 'Simple' model on test data
myStats.get_score_report(best_fit_tree, tree_y_test, tree_X_test)

# %%
# 'Advanced' model on test data
myStats.get_score_report(best_fit_tree_adv, tree_y_test, tree_X_test_adv)

# The model 'advanced' is slightly slightly better

# %%
# Precision-Recall Plot

myPlots.plot_precision_recall(
    list_classifier = [best_fit_tree, best_fit_tree_adv],
    list_X = [tree_X_test, tree_X_test_adv],
    true_y = tree_y_test,
    list_names = ['Simple Model', 'Adv Model'], 
    pos_label = 1
)

# %% 
# Check Models feature importance 
myStats.get_feature_importance(
    classifier = best_fit_tree,
    col_names = tree_X_test.columns
    )

# %% 
# Check Models feature importance 
myStats.get_feature_importance(
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