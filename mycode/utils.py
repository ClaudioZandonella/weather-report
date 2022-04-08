#===============================#
#====    Utils Functions    ====#
#===============================#

import shelve
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

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

#----    get_encoded_data    ----#

def get_encoded_data(df_train, df_test, categ_var, numeric_var):
    '''
    return X, X_test, y_test with selected numeric column and 
    categorical columns encoded with OneHotEncoder
    '''
    # Create encoder
    var_encoder = OneHotEncoder(categories='auto', sparse = False)
    var_encoder.fit(df_test[categ_var])

    encoded_columns = var_encoder.get_feature_names_out(categ_var)

    # Get encoded categorical data
    encoded_data = var_encoder.transform(df_train[categ_var])
    encoded_data = pd.DataFrame(encoded_data, columns = encoded_columns, index = df_train.index)

    # Train data
    X = pd.concat([df_train[numeric_var], encoded_data], axis=1)

    # Encode test data

    # Get encoded categorical data test
    encoded_data_test = var_encoder.transform(df_test[categ_var])
    encoded_data_test = pd.DataFrame(encoded_data_test, columns = encoded_columns, index = df_test.index)

    # Test data
    X_test = pd.concat([df_test[numeric_var], encoded_data_test], axis=1)

    return (X, X_test)

