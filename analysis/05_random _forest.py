#================================#
#====    05 Random Forest    ====#
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

import sys, os
import shelve
from time import time
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'mycode'))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  PrecisionRecallDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# from mycode
import utils  
import myStats
import myPlots

# get data

d_old = shelve.open(os.path.join(os.path.dirname(sys.path[0]),'outputs', '04_decision_tree'))
# Training Data
forest_X = d_old['tree_X'].copy()
forest_X_adv = d_old['tree_X_adv'].copy()
forest_y = d_old['tree_y'].copy()

# Test Data
forest_X_test = d_old['tree_X_test'].copy()
forest_X_test_adv = d_old['tree_X_test_adv'].copy()
forest_y_test = d_old['tree_y_test'].copy()

d_old.close()

#%%
#----    01 Encode Data    ----#

# Same variables as in the decision tree analysis

# Training data
print(forest_X.shape)
print(forest_y.shape)

# %%
# Test data
print(forest_X_test.shape)
print(forest_y_test.shape)

# %%
##%%time

#----    02 Fit Model    ----#

# First trial random forest

fit_forest = RandomForestClassifier(
    n_estimators = 100, criterion = 'gini',
    max_depth = 10, max_features = "sqrt",
    bootstrap = True, n_jobs = 6,
    class_weight = 'balanced', random_state = 2022)\
    .fit(forest_X, forest_y)

#%%
# Get info score including confsion matrix, classification report, and f2 values
myStats.get_score_report(fit_forest, forest_y, forest_X)

# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    fit_forest, forest_X, forest_y, name="Initial Model"
)

#%%
features = myStats.get_feature_importance(fit_forest, forest_X.columns)
features

# %%
sns.barplot(data = features.head(10), x = 'importance', y = 'feature')

#%%

#----    03 Grid Search    ----#

ensemble_forests = myStats.get_grid_forests(
    class_weights = [{0:1, 1:3.5}],
    max_features = [10, 20],
    max_depth = [5, 10, 15],
    random_state = 2022
    )

ensemble_forests_II = myStats.get_grid_forests(
    class_weights = [{0:1, 1:3.5}],
    max_features = [10, 20],
    max_depth = [8, 10, 12],
    random_state = 2022
    )

ensemble_forests_III = myStats.get_grid_forests(
    class_weights = [{0:1, 1:3.5}, {0:1, 1:5}, {0:1, 1:7}],
    max_features = [20, 40],
    max_depth = [10],
    random_state = 2022
    )

ensemble_forests_IV = myStats.get_grid_forests(
    class_weights=[{0:1, 1:6}, {0:1, 1:7}, {0:1, 1:8}],
    max_features= [40, 60],
    max_depth = [10],
    random_state=2022
    )

ensemble_forests_V = myStats.get_grid_forests(
    class_weights=[{0:1, 1:7}],
    max_features= [40],
    max_depth = [10, 11, 12],
    random_state=2022
    )

ensemble_forests_VI = myStats.get_grid_forests( # to allow fast compilation
    class_weights=[{0:1, 1:7}],
    max_features= [40],
    max_depth = [11],
    random_state=2022
    )

#%%

## %%time
# No  big improvements after 350
my_range = range(360, 400 + 1, 20)
oobf2_rates = myStats.get_oobf2_rates(ensemble_forests_VI, my_range, forest_X, forest_y)

# %%

myPlots.plot_oobf2(oobf2_rates)

# %%

## %%time
#----    04 Best Model    ----#

# 'max_depth': 11;    'class_weight': {0:1, 1:7}; 'max_features' = 40
best_fit_forest = RandomForestClassifier(
    n_estimators = 350, criterion = 'gini',
    max_depth = 11, max_features = 40,
    bootstrap = True, n_jobs = 6,
    class_weight = {0:1, 1:7}, random_state = 2022)\
    .fit(forest_X, forest_y)

#%%
# Get info score including confsion matrix, classification report, and f2 values
myStats.get_score_report(best_fit_forest, forest_y, forest_X)

# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    best_fit_forest, forest_X, forest_y, name="Best Model"
)

#%%
best_features = myStats.get_feature_importance(best_fit_forest, forest_X.columns)
best_features

# %%
sns.barplot(data = best_features.head(10), x = 'importance', y = 'feature')

# %%

#----    05 Encode Data Advanced    ----#

# Same variables as in the decision tree analysis

# Training data
print(forest_X_adv.shape)
print(forest_y.shape)

# %%
# Test data
print(forest_X_test_adv.shape)
print(forest_y_test.shape)

# %%
## %%time

#----    06 Fit Model Advanced    ----#

# First trial random forest

fit_forest_adv = RandomForestClassifier(
    n_estimators = 100, criterion = 'gini',
    max_depth = 10, max_features = "sqrt",
    bootstrap = True, n_jobs = 6,
    class_weight = 'balanced', random_state = 2022)\
    .fit(forest_X_adv, forest_y)

#%%
# Get info score including confsion matrix, classification report, and f2 values
myStats.get_score_report(fit_forest_adv, forest_y, forest_X_adv)

# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    fit_forest_adv, forest_X_adv, forest_y, name="Initial Model"
)

#%%
# Check feature
features_adv = myStats.get_feature_importance(fit_forest_adv, forest_X_adv.columns)
features_adv

# %%
# Plot features
sns.barplot(data = features_adv.head(10), x = 'importance', y = 'feature')

#%%

#----    07 Grid Search Advanced    ----#

ensemble_forests = myStats.get_grid_forests(
    class_weights = [{0:1, 1:7}],
    max_features = [10, 20],
    max_depth = [5, 10, 15],
    random_state = 2022
    )

ensemble_forests_II = myStats.get_grid_forests(
    class_weights = [{0:1, 1:7}],
    max_features = [10],
    max_depth = [14, 15, 16],
    random_state = 2022
    )

ensemble_forests_III = myStats.get_grid_forests(
    class_weights = [{0:1, 1:6}, {0:1, 1:7}, {0:1, 1:8}],
    max_features = [10],
    max_depth = [15],
    random_state = 2022
    )

ensemble_forests_IV = myStats.get_grid_forests(
    class_weights = [{0:1, 1:7}],
    max_features = [20, 30, 40],
    max_depth = [10],
    random_state = 2022
    )

ensemble_forests_V = myStats.get_grid_forests(
    class_weights = [{0:1, 1:7}],
    max_features = [40],
    max_depth = [10, 12, 14],
    random_state = 2022
    )

ensemble_forests_VI = myStats.get_grid_forests( # to allow fast compilation
    class_weights = [{0:1, 1:7}],
    max_features = [40],
    max_depth = [12],
    random_state = 2022
    )
#%%

## %%time
# No  big improvements after 350
my_range = range(360, 400 + 1, 20)
oobf2_rates = myStats.get_oobf2_rates(ensemble_forests_VI, my_range, forest_X, forest_y)

# %%

myPlots.plot_oobf2(oobf2_rates)

# %%

## %%time
#----    08 Best Model    ----#

# 'max_depth': 12;    'class_weight': {0:1, 1:7}; 'max_features' = 40
best_fit_forest_adv = RandomForestClassifier(
    n_estimators = 350, criterion = 'gini',
    max_depth = 12, max_features = 40,
    bootstrap = True, n_jobs = 6,
    class_weight = {0:1, 1:7}, random_state = 2022)\
    .fit(forest_X_adv, forest_y)

#%%
# Get info score including confsion matrix, classification report, and f2 values
myStats.get_score_report(best_fit_forest_adv, forest_y, forest_X_adv)

# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    best_fit_forest_adv, forest_X_adv, forest_y, name = "Best Model"
)

#%%
# Check Features
best_features_adv = myStats.get_feature_importance(best_fit_forest_adv, forest_X_adv.columns)
best_features_adv

# %%
# Plot Features
sns.barplot(data = best_features_adv.head(10), x = 'importance', y = 'feature')

#%%
#----    09 Models Comparison    ----#

# 'Simple' model on test data
myStats.get_score_report(best_fit_forest, forest_y_test, forest_X_test)

# %%
# 'Advanced' model on test data
myStats.get_score_report(best_fit_forest_adv, forest_y_test, forest_X_test_adv)

# The model 'advanced' is slightly slightly better

# %%
# Precision-Recall Plot

myPlots.plot_precision_recall(
    list_classifier = [best_fit_forest, best_fit_forest_adv],
    list_X = [forest_X_test, forest_X_test_adv],
    true_y = forest_y_test,
    list_names = ['Simple Model', 'Adv Model'], 
    pos_label = 1
)

# %% 
# Check Models feature importance 
myStats.get_feature_importance(
    classifier = best_fit_forest,
    col_names = forest_X_test.columns
    )

# %% 
# Check Models feature importance 
myStats.get_feature_importance(
    classifier = best_fit_forest_adv,
    col_names = forest_X_test_adv.columns
    )

# %%

#----    10 End Forest    ----#

# Save Objects
d = shelve.open(os.path.join(os.path.dirname(sys.path[0]),'outputs', '05_random_forest'))

# Training Data
d['forest_X'] = forest_X
d['forest_X_adv'] = forest_X_adv
d['forest_y'] = forest_y

# Test Data
d['forest_X_test'] = forest_X_test
d['forest_X_test_adv'] = forest_X_test_adv
d['forest_y_test'] = forest_y_test

# Models
d['best_fit_forest'] = best_fit_forest
d['best_fit_forest_adv'] = best_fit_forest_adv
d.close()