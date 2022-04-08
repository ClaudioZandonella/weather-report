#===============================#
#====    Utils Functions    ====#
#===============================#

import shelve
import numpy as np
import pandas as pd
import itertools
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
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

#----    get_coef_importance    ----#

def get_coef_importance(classifier, col_names):
    '''
    Given the clasisfier return dict of coef ordered for importance (absolute value)
    '''

    old_shape = classifier.coef_.shape
    coeff_val = classifier.coef_.reshape(old_shape[1])
    coeff = pd.DataFrame({
        'coefficient' : col_names,
        'value' : coeff_val
    }).sort_values('value', key = lambda col: abs(col), ascending=False)
    
    return coeff

#----    get_feature_importance    ----#

def get_feature_importance(classifier, col_names):
    '''
    Given the clasisfier return dict of feaatures ordered for importance (absolute value)
    '''

    feat = pd.DataFrame({
        'feature' : col_names,
        'importance' : classifier.feature_importances_
    }).sort_values('importance', ascending=False)

    return feat

#----    get_oobf2_rates    ----

def get_oobf2_rates(ensemble_forests, my_range, X, y):
    '''
    compute oob f2 value for a range of trees for each clissifier
    '''

    f2_rate = OrderedDict((label, []) for label, _ in ensemble_forests)

    for label, clf in ensemble_forests:
        print(label)
        for i in my_range:
            if(i == my_range[-1]):
                print(i)
            else:
                print(i, end='  ')
            
            if(i == my_range[0]): # run in parallell the firs step
                n_jobs = 6
                warm_start=False
            else:                 # update subsequent steps
                n_jobs = 1
                warm_start=True

            clf.set_params(
                n_estimators=i,
                n_jobs = n_jobs,
                warm_start = warm_start)
                
            clf.fit(X, y)

            # Record the OOB f2 for each `n_estimators=i` setting.
            pred_y = np.argmax(clf.oob_decision_function_, axis=1)
            f2_score = fbeta_score(y, pred_y, pos_label = 1, beta = 2)   
            f2_rate[label].append((i, f2_score))
    
    return f2_rate


#----    get_grid_forests    ----#
def get_grid_forests(
    class_weights, 
    max_features,
    max_depth,
    random_state):
    '''
    create a list of classifiers combining the possible values of 
    class_weights, max_features, max_depth
    '''

    combinations = list(itertools.product(class_weights, max_features, max_depth))

    ensemble_forests = []
    for i in combinations:
        forest = (
            f"class_weight = {i[0]}; max_features={i[1]}; max_depth={i[2]}",
            RandomForestClassifier(
                oob_score=True,
                class_weight = i[0],
                max_features = i[1],
                max_depth = i[2],
                random_state = random_state,
                )
            )
        ensemble_forests.append(forest)

    return ensemble_forests
