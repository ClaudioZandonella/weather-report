#===============================#
#====    Utils Functions    ====#
#===============================#

import shelve
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, fbeta_score

#----    my_count    ----

def my_count(x, sort_index = True):
    '''
    get count and proportion from series
    '''
    count = x.value_counts(dropna = False)
    prop = x.value_counts(dropna = False, normalize = True)

    res = pd.DataFrame({'count':count, 
                        'prop':prop})
    
    if sort_index:
        res.sort_index(inplace=True)

    return res

'''
x = pd.Series(['a', 'b', 'b', 'b', 'a'])
my_count(x, sort_index = True)
'''

#----    get_days_diff    ----#

def get_days_diff(data, periods = 1, col_name = 'Date', group_by = 'Location'):
    '''
    Get days differene between observations (absolute value). 
    Periods is used to define the shift for calculating difference
    '''

    days_diff = data.groupby(group_by)[col_name]\
        .diff(periods = periods).abs() / np.timedelta64(1, 'D')
    days_diff.fillna(0, inplace = True)

    return days_diff

'''
d_old = shelve.open("outputs/01_data_explore")
zz = d_old['df'].copy()
d_old.close()
get_days_diff(zz, periods = -1)
'''

#----    smart_fillna    ----#

def smart_fillna(data, column):
    '''
    Substitute NaN with the mean yesterday and tomorrow values  
    '''
    nan_index = data[data[column].isna()].index

    for i in nan_index:
        # Test if there are values for yesterday and tomorrow (up to two days difference)
        my_test = (data.loc[i]['DiffPrev'] <= 2) & (data.loc[i]['DiffNext'] <= 2) & (data.loc[i]['DiffPrev'] != 0) & (data.loc[i]['DiffNext'] != 0) 
        
        # skip dates that have no yesterday and tomorrow
        if my_test == False:  # my_test is False do not work why?!?
            continue

        # Compute today value as mean yesterday and tomorrow values
        yesterday_val = data.loc[i-1][column]
        tomorrow_val = data.loc[i+1][column]

        today_val = np.mean([yesterday_val, tomorrow_val])

        data.loc[i, column] = today_val
    

'''
import shelve


# get data
d_old = shelve.open("outputs/01_data_explore")
zz = d_old['df'].copy()
d_old.close()

my_index = zz[zz['MinTemp'].isna()].index
zz.loc[my_index]

smart_fillna(zz, 'MinTemp')
'''

#----    get_fbeta_score    ----

def get_fbeta_score(true_y, pred_y):
    '''
    get values of f2 for clas 0, class 1, and average (macro and weighted)
    '''

    values = {
        'fbeta0' : fbeta_score(true_y, pred_y, pos_label = 0, beta = 2),
        'fbeta1' : fbeta_score(true_y, pred_y, pos_label = 1, beta = 2),
        'fbeta_macro' : fbeta_score(true_y, pred_y, average = 'macro', beta = 2),
        'fbeta_weight' : fbeta_score(true_y, pred_y, average = 'weighted', beta = 2),
    }
    
    
    return pd.DataFrame(values, index = ['0'])



#----    get_score_report    ----#

def get_score_report(classifier, true_y, true_X):
    '''
    Get info score including confsion matrix, classification report, and f2 values
    '''

    y_pred = classifier.predict(true_X)
    print(confusion_matrix(true_y, y_pred))
    print(classification_report(true_y, y_pred))
    print(get_fbeta_score(true_y, y_pred))

    return None
