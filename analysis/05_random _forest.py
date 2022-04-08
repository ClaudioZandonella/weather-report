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

import sys
import shelve
from time import time
sys.path.append('../mycode')

import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy import stats 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, fbeta_score,\
    PrecisionRecallDisplay, make_scorer, precision_recall_curve
from sklearn.model_selection import KFold, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

import utils  # from mycode
import myPlots  # from mycode

# get data

d_old = shelve.open("../outputs/04_decision_tree")
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
%%time

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
utils.get_score_report(fit_forest, forest_y, forest_X)

# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    fit_forest, forest_X, forest_y, name="Initial Model"
)

#%%
features = utils.get_feature_importance(fit_forest, forest_X.columns)
features

# %%
sns.barplot(data = features.head(10), x = 'importance', y = 'feature')

#%%

#----    03 Grid Search    ----#

ensemble_forests = utils.get_grid_forests(
    class_weights=[{0:1, 1:3.5}],
    max_features= [10, 20],
    max_depth = [5, 10, 15],
    random_state=2022
    )

ensemble_forests_II = utils.get_grid_forests(
    class_weights=[{0:1, 1:3.5}],
    max_features= [10, 20],
    max_depth = [8, 10, 12],
    random_state=2022
    )

ensemble_forests_III = utils.get_grid_forests(
    class_weights=[{0:1, 1:3.5}, {0:1, 1:5}, {0:1, 1:7}],
    max_features= [20, 40],
    max_depth = [10],
    random_state=2022
    )

ensemble_forests_IV = utils.get_grid_forests(
    class_weights=[{0:1, 1:6}, {0:1, 1:7}, {0:1, 1:8}],
    max_features= [40, 60],
    max_depth = [10],
    random_state=2022
    )

ensemble_forests_V = utils.get_grid_forests(
    class_weights=[{0:1, 1:7}],
    max_features= [40],
    max_depth = [10, 11, 12],
    random_state=2022
    )

#%%

%%time
# No  big improvements after 350
my_range = range(360, 400 + 1, 20)
oobf2_rates= utils.get_oobf2_rates(ensemble_forests_V, my_range, forest_X, forest_y)

# %%

myPlots.plot_oobf2(oobf2_rates)

# %%

%%time
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
utils.get_score_report(best_fit_forest, forest_y, forest_X)

# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    best_fit_forest, forest_X, forest_y, name="Best Model"
)

#%%
best_features = utils.get_feature_importance(best_fit_forest, forest_X.columns)
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
%%time

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
utils.get_score_report(fit_forest_adv, forest_y, forest_X_adv)

# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    fit_forest_adv, forest_X_adv, forest_y, name="Initial Model"
)

#%%
features_adv = utils.get_feature_importance(fit_forest_adv, forest_X_adv.columns)
features_adv

# %%
sns.barplot(data = features_adv.head(10), x = 'importance', y = 'feature')

#%%

#----    07 Grid Search Advancedd    ----#

ensemble_forests = utils.get_grid_forests(
    class_weights=[{0:1, 1:7}],
    max_features= [10, 20],
    max_depth = [5, 10, 15],
    random_state=2022
    )

ensemble_forests_II = utils.get_grid_forests(
    class_weights=[{0:1, 1:7}],
    max_features= [10],
    max_depth = [14, 15, 16],
    random_state=2022
    )

ensemble_forests_III = utils.get_grid_forests(
    class_weights=[{0:1, 1:6}, {0:1, 1:7}, {0:1, 1:8}],
    max_features= [10],
    max_depth = [15],
    random_state=2022
    )

ensemble_forests_IV = utils.get_grid_forests(
    class_weights=[{0:1, 1:7}],
    max_features= [20, 30, 40],
    max_depth = [10],
    random_state=2022
    )

ensemble_forests_V = utils.get_grid_forests(
    class_weights=[{0:1, 1:7}],
    max_features= [40],
    max_depth = [10, 12, 14],
    random_state=2022
    )

#%%

%%time
# No  big improvements after 350
my_range = range(360, 400 + 1, 20)
oobf2_rates= utils.get_oobf2_rates(ensemble_forests_V, my_range, forest_X, forest_y)

# %%

myPlots.plot_oobf2(oobf2_rates)

# %%

%%time
#----    04 Best Model    ----#

# 'max_depth': 12;    'class_weight': {0:1, 1:7}; 'max_features' = 40
best_fit_forest_adv = RandomForestClassifier(
    n_estimators = 350, criterion = 'gini',
    max_depth = 12, max_features = 40,
    bootstrap = True, n_jobs = 6,
    class_weight = {0:1, 1:7}, random_state = 2022)\
    .fit(forest_X_adv, forest_y)

#%%
# Get info score including confsion matrix, classification report, and f2 values
utils.get_score_report(best_fit_forest_adv, forest_y, forest_X_adv)

# %%
# Precision-Recall Plot
display = PrecisionRecallDisplay.from_estimator(
    best_fit_forest_adv, forest_X_adv, forest_y, name="Best Model"
)

#%%
best_features_adv = utils.get_feature_importance(best_fit_forest_adv, forest_X_adv.columns)
best_features_adv

# %%
sns.barplot(data = best_features_adv.head(10), x = 'importance', y = 'feature')

#%%
#----    09 Models Comparison    ----#

# 'Simple' model on test data
utils.get_score_report(best_fit_forest, forest_y_test, forest_X_test)

# %%
# 'Advanced' model on test data
utils.get_score_report(best_fit_forest_adv, forest_y_test, forest_X_test_adv)

# The model 'advanced' is slightly slightly bettter

# %%
# Precision-Recall Plot

myPlots.plot_precision_recall(
    list_classifier = [best_fit_forest, best_fit_forest_adv],
    list_X =[forest_X_test, forest_X_test_adv],
    true_y = forest_y_test,
    list_names = ['Simple Model', 'Adv Model'], 
    pos_label = 1
)

# %% 
# Check Models feature importance 
utils.get_feature_importance(
    classifier = best_fit_forest,
    col_names = forest_X_test.columns
    )

# %% 
# Check Models feature importance 
utils.get_feature_importance(
    classifier = best_fit_forest_adv,
    col_names = forest_X_test_adv.columns
    )

# %%

#----    10 End Forest    ----#

# Save Objects
d = shelve.open("../outputs/05_random_forest")

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