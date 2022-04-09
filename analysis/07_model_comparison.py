#===================================#
#====    07 Model Comparison    ====#
#===================================#

# 01 Load Models
# 02 Compare Models
# 03 Grid Search


#%%

#----    Settings    ----#

import sys
import shelve
from time import time
sys.path.append('../mycode')

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from mycode
import utils 
import myStats 
import myPlots 

# %%
#----    01 Load Models    ----#

# Logistic Regression

d_old = shelve.open("../outputs/03_logistic_reg")

# Test Data
logistic_X_test = d_old['logistic_X_test'].copy()
logistic_X_test_adv = d_old['logistic_X_test_adv'].copy()
y_test = d_old['logistic_y_test'].copy()

# Models
fit_logistic = d_old['best_fit_logistic']
fit_logistic_adv = d_old['best_fit_logistic_adv']

d_old.close()

#%%
# Decsion Tree

d_old = shelve.open("../outputs/04_decision_tree")

# Test Data
tree_X_test = d_old['tree_X_test'].copy()
tree_X_test_adv = d_old['tree_X_test_adv'].copy()

# Models
fit_tree = d_old['best_fit_tree']
fit_tree_adv = d_old['best_fit_tree_adv']

d_old.close()

#%%
# Random Forest

d_old = shelve.open("../outputs/05_random_forest")

# Data are the same of tree models

# Models
fit_forest = d_old['best_fit_forest']
fit_forest_adv = d_old['best_fit_forest_adv']

d_old.close()

#%%
# XGBoost

d_old = shelve.open("../outputs/06_xgboost")

# Test Data
xgb_X_test = d_old['xgb_X_test'].copy()
xgb_X_test_adv = d_old['xgb_X_test_adv'].copy()

# Models
fit_xgb = d_old['best_fit_xgb']
fit_xgb_adv = d_old['best_fit_xgb_adv']

d_old.close()

#%%

#----    02 Model Comparison    ----#

# %%
# Logistic regression
myStats.get_score_report(fit_logistic, y_test, logistic_X_test)

# %%
# Decision tree
myStats.get_score_report(fit_tree, y_test, tree_X_test)

# %%
# Random forest
myStats.get_score_report(fit_forest, y_test, tree_X_test)

# %%
# XGBoost
myStats.get_score_report(fit_xgb, y_test, xgb_X_test)

# %%

# Precision-Recall Plot
myPlots.plot_precision_recall(
    list_classifier = [fit_logistic, fit_tree, fit_forest, fit_xgb],
    list_X = [logistic_X_test, tree_X_test, tree_X_test, xgb_X_test],
    true_y = y_test,
    list_names = ['Logistic', 'Tree', 'Forest', 'XGB'], 
    pos_label = 1
)
# %%

#----    02 Model Comparison Advanced    ----#

# %%
# Logistic regression
myStats.get_score_report(fit_logistic_adv, y_test, logistic_X_test_adv)

# %%
# Decision tree
myStats.get_score_report(fit_tree_adv, y_test, tree_X_test_adv)

# %%
# Random forest
myStats.get_score_report(fit_forest_adv, y_test, tree_X_test_adv)

# %%
# XGBoost
myStats.get_score_report(fit_xgb_adv, y_test, xgb_X_test_adv)

# %%

# Precision-Recall Plot
myPlots.plot_precision_recall(
    list_classifier = [fit_logistic_adv, fit_tree_adv, fit_forest_adv, fit_xgb_adv],
    list_X = [logistic_X_test_adv, tree_X_test_adv, tree_X_test_adv, xgb_X_test_adv],
    true_y = y_test,
    list_names = ['Logistic', 'Tree', 'Forest', 'XGB'], 
    pos_label = 1
)
# %%

