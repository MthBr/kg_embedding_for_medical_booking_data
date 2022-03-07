#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:27:46 2019

@author: modal
"""

def save_on_pickle(df,file_name, verbose=0):
    if verbose:
        print("Saving dataframe in the pickle file '{}' ...".format(file_name.split('/')[-1]))
    import pickle
    pickling_on = open(file_name,"wb")
    pickle.dump(df, pickling_on, protocol=4)
    pickling_on.close()
    
def load_pickle(file_name, verbose=0):
    if verbose:
        print("Loading the pickle file '{}' ...".format(file_name.split('/')[-1]))
    import pickle
    pickle_off = open(file_name,"rb")
    df = pickle.load(pickle_off)
    return df

def scale_01_allaxis(df_X, range=(0,1), minval=None, maxval=None):
    mi, ma = range
    X_min = minval if minval != None else df_X.min().min()
    X_max = maxval if maxval != None else df_X.max().max()
    df_X_std = df_X - X_min
    if X_max != X_min:
        df_X_std = df_X_std / (X_max - X_min)
    df_X_scaled = df_X_std * (ma - mi) + mi
    return df_X_scaled

def get_dummies_ord(df, drop_first=False):
    import pandas as pd
    return pd.concat([pd.get_dummies(df[col], prefix=col, drop_first=drop_first) if df[col].dtype == object else df[col] for col in df], axis=1)

def get_df_and_values_from_cols(df, cols):
    df_res = df[cols]
    res = df_res.values
    return df_res, res

def get_trainvaltest_df_and_values_from_cols(df_train, df_val, df_test, cols):
    df_train_res_in, train_res_in = get_df_and_values_from_cols(df_train, cols)
    df_val_res_in, val_res_in = get_df_and_values_from_cols(df_val, cols)
    df_test_res_in, test_res_in = get_df_and_values_from_cols(df_test, cols)

    return df_train_res_in, train_res_in, \
        df_val_res_in, val_res_in, \
        df_test_res_in, test_res_in

def convert_01(data, threshold=0.5):
    import pandas as pd
    s_data = pd.Series(data, copy=True)
    s_data[s_data < threshold] = 0
    s_data[s_data >= threshold] = 1
    return s_data

# demonstration of calculating metrics for a neural network model using sklearn
def show_stats(df_real, df_pred, df_pred_prob):
    from sklearn.metrics import accuracy_score
#    from sklearn.metrics import precision_score
#    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
#    from sklearn.metrics import cohen_kappa_score
#    from sklearn.metrics import roc_auc_score
#    from sklearn.metrics import confusion_matrix
    
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(df_real, df_pred)
    print('Accuracy: %f' % accuracy)
#    # precision tp / (tp + fp)
#    precision = precision_score(df_real, df_pred)
#    print('Precision: %f' % precision)
#    # recall: tp / (tp + fn)
#    recall = recall_score(df_real, df_pred)
#    print('Recall: %f' % recall)
#    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(df_real, df_pred)
    print('F1 score: %f' % f1)
#    
#    # kappa
#    kappa = cohen_kappa_score(df_real, df_pred)
#    print('Cohens kappa: %f' % kappa)
#    # ROC AUC
#    auc = roc_auc_score(df_real, df_pred_prob)
#    print('ROC AUC: %f' % auc)
#    # confusion matrix
#    matrix = confusion_matrix(df_real, df_pred_prob)
#    print(matrix)