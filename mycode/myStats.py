#========================#
#====    My Stats    ====#
#========================#

import numpy as np
import pandas as pd
import itertools
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, fbeta_score


#----    get_fbeta_score    ----

def get_fbeta_score(true_y, pred_y):
    '''
    get values of f2 for class 0, class 1, and average (macro and weighted)
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
    index_matrix = pd.MultiIndex.from_tuples([('True_y', 0), ('True_y', 1)])
    column_matrix = pd.MultiIndex.from_tuples([('Pred_y', 0), ('pred_y', 1)])
    matrix = confusion_matrix(true_y, y_pred)

    print('Confusion Matrix')
    print(pd.DataFrame(matrix, index = index_matrix, columns=column_matrix))
    print('\nClassification Report')
    print(classification_report(true_y, y_pred))
    print('\nF2 Score')
    print(get_fbeta_score(true_y, y_pred))

    return None

#----    get_coef_importance    ----#

def get_coef_importance(classifier, col_names):
    '''
    Given the classifier return dict of coef ordered for importance (absolute value)
    '''

    old_shape = classifier.coef_.shape
    coef_val = classifier.coef_.reshape(old_shape[1])
    coef = pd.DataFrame({
        'coefficient' : col_names,
        'value' : coef_val
    }).sort_values('value', key = lambda col: abs(col), ascending=False)
    
    return coef

#----    get_feature_importance    ----#

def get_feature_importance(classifier, col_names):
    '''
    Given the classifier return dict of features ordered for importance (absolute value)
    '''

    feat = pd.DataFrame({
        'feature' : col_names,
        'importance' : classifier.feature_importances_
    }).sort_values('importance', ascending=False)

    return feat

#----    get_oobf2_rates    ----

def get_oobf2_rates(ensemble_forests, my_range, X, y):
    '''
    compute oob f2 value for a range of trees for each classifier
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
