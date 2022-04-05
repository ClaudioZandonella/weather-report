#==============================#
#====    Plot Functions    ====#
#==============================#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

