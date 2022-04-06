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
    Plot histogram of continous variables
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
        y_pred = list_classifier[i].predict(list_X[i])
        f2_score = fbeta_score(true_y, y_pred, pos_label = pos_label, beta = 2)
        precision, recall, _ = precision_recall_curve(true_y, y_pred_prob[:, pos_label])

        display = PrecisionRecallDisplay(
            recall=recall,
            precision=precision
        )
        display.plot(ax=ax, name=f"Precision-recall for {list_names[i]} (F2 {f2_score:.2f})")

    ax.set_xlabel(f'Recall (Positive label: {pos_label})')
    ax.set_ylabel(f'Precision (Positive label: {pos_label})')


