#==============================#
#====    Plot Functions    ====#
#==============================#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score,\
    PrecisionRecallDisplay, fbeta_score


#----    plot_descriptive    ----#

def plot_descriptive(data, columns, outcome = None):
    '''
    Plot histogram of continuos variables
    '''

    n_var = len(columns)
    n_row = (n_var + 1)//2

    fig, axes = plt.subplots(n_row,2, figsize= (14,n_row*4))

    for i, col_name in enumerate(columns):
        if outcome is not None:
            data.pivot(columns=outcome, values=col_name).plot.hist(ax=axes[i%n_row][i//n_row], alpha=0.6, bins = 30)
        else:
            data[col_name].hist(ax=axes[i%n_row][i//n_row], alpha=0.7, bins = 30);
        axes[i%n_row][i//n_row].set_title(f'{col_name}', fontsize=13);
        plt.subplots_adjust(hspace=0.45)


#----    plot_precision_recall    ----#

def plot_precision_recall(list_classifier, list_X, true_y, list_names, pos_label = 1):

    _, ax = plt.subplots(figsize=(7, 5))

    for i in range(len(list_classifier)):

        y_pred_prob = list_classifier[i].predict_proba(list_X[i])
        ap_score = average_precision_score(true_y, y_pred_prob[:, pos_label])
        precision, recall, _ = precision_recall_curve(true_y, y_pred_prob[:, pos_label])

        display = PrecisionRecallDisplay(
            recall = recall,
            precision = precision,
            average_precision = ap_score
        )
        display.plot(ax=ax, name=f"Precision-recall for {list_names[i]}")

    ax.set_xlabel(f'Recall (Positive label: {pos_label})')
    ax.set_ylabel(f'Precision (Positive label: {pos_label})')

#----    plot_grid_tree    ----

def plot_grid_tree(grid_tree):
    class_weights = grid_tree['param_class_weight'].value_counts().index
    depths = sorted(grid_tree['param_max_depth'].unique())

    fig, ax = plt.subplots(1,1, figsize=(15,5))
    for i in class_weights:
        cv_scores_mean = grid_tree[grid_tree['param_class_weight'] == i].\
            sort_values(by=['param_max_depth'])['mean_test_score']
        cv_scores_std = grid_tree[grid_tree['param_class_weight'] == i].\
            sort_values(by=['param_max_depth'])['std_test_score']
        
        ax.plot(depths, cv_scores_mean, '-o', label=f'Class Weights: {i}', alpha=0.9)
        ax.fill_between(depths, cv_scores_mean-2*cv_scores_std, cv_scores_mean+2*cv_scores_std, alpha=0.1)
    ax.set_xlabel('Tree depth')
    ax.set_ylabel('F2 (Positive label: 1)')
    ax.legend()

#----    plot_oobf2    ----#

def plot_oobf2(f2_rates):
    # Generate the OOB F2 vs. number of trees
    _, ax = plt.subplots(figsize=(7, 5))
    
    for label, clf_f2 in f2_rates.items():
        xs, ys = zip(*clf_f2)
        plt.plot(xs, ys, label=label)

    plt.xlabel("n trees")
    plt.ylabel("F2")
    plt.legend(loc="lower right")
    plt.show()

