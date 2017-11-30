#!/usr/bin/env python

import numpy as np
import pandas as pd
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#import xgboost as xgb

##make sure to have installed skilearn version 0.19
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import CondensedNearestNeighbour, RandomUnderSampler


def read_datasets(list_data = ['final_txn_v2', 'final_user_log_v2']):
    for i in range(0, len(list_data)):
        temp_df = pd.read_csv('../'+list_data[i]+'.csv')
        if i == 0:
            merged_df = temp_df
        else:
            merged_df = merged_df.merge(temp_df, on='new_id', how='inner')

    return merged_df

def prep_variables(merged_df):
    train=merged_df
    #DUMMIES FOR MONTHS_LISTENING
    train = train.join(pd.get_dummies(train['months_listening'])).drop('months_listening', axis = 1)

    #DUMMIES FOR CHURN
    train = train.join(pd.get_dummies(train['cluster9'], prefix = 'cluster_')).drop('cluster9', axis = 1)

    #DUMMIES FOR REGISTERED_VIA
    train = train.join(pd.get_dummies(train['registered_via'], prefix = 'reg_via_')).drop('registered_via', axis = 1)

    #Fix LAST_CANCEL string issue
    train['last_cancel'] = train['last_cancel'].apply(pd.to_numeric, errors='coerce').fillna(0)

    #TURN 'NEW_ID' INTO INDEX
    train = train.set_index('new_id')

    return train

def oversample(train):
    y = train['is_churn']
    x = train.drop('is_churn', axis = 1)
    print ('Oversampling [SMOTE]...')
    sm = SMOTE(random_state=27) #check for parameters tunning
    x_res, y_res = sm.fit_sample(x, y)
    print ('Finished oversampling')
    return x_res, y_res

def undersample(train):
    y = train['is_churn']
    x = train.drop('is_churn', axis = 1)
    print ('Undersampling [Random]...')
    rus = RandomUnderSampler(random_state=27, replacement=True) #check for parameters tunning
    x_res, y_res = rus.fit_sample(x, y)
    print ('Finished undersampling')
    return x_res, y_res

def train_test(train, test_size = 0.20, seed = 27, oversampling = 0):
    data_model = {}

    if oversampling == 1:
        x, y = oversample(train)
    elif oversampling ==-1:
        x, y = undersample(train)
    else:
        y = train['is_churn']
        x = train.drop(['is_churn'], axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = seed)

    return X_train, X_test, y_train, y_test

def printm(y_train, y_prob_train, y_test, y_prob):
    print('Train:\t{:.4f}' .format(log_loss(y_train, y_prob_train)))
    print('Test:\t{:.4f}' .format(log_loss(y_test, y_prob)))

def model_forest(X_train, X_test, y_train, y_test, logprint=1, trees=25, criterion = 'gini', max_depth=25):
    forest = RandomForestClassifier(n_estimators= trees, criterion = criterion, max_depth=max_depth).fit(X_train, y_train)
    y_prob = forest.predict_proba(X_test)
    y_prob_train = forest.predict_proba(X_train)
    [print ('Trees: {}, Criterion: {}, Max_depth: {}' .format(trees, criterion, max_depth)) if logprint == 1 else False]
    printm(y_train, y_prob_train, y_test, y_prob)

def model_svm(X_train, X_test, y_train, y_test, logprint=1, kernel='sigmoid', C = 10, gamma=0.1, probability=True):
    svm =  SVC(kernel = kernel, C=C, gamma=gamma).fit(X_train, y_train)
    y_prob = svm.predict_proba(X_test)
    y_prob_train = svm.predict_proba(X_train)
    print ('kernel: {}, C: {}, gamma: {}' .format(kernel, C, gamma))
    printm(y_train, y_prob_train, y_test, y_prob)

def model_gaussian(X_train, X_test, y_train, y_test, prior = None):
    gau =  GaussianNB().fit(X_train, y_train)
    y_prob = gau.predict_proba(X_test)
    y_prob_train = gau.predict_proba(X_train)
    print ('Prior: {}' .format(prior))
    printm(y_train, y_prob_train, y_test, y_prob)

def model_log(X_train, X_test, y_train, y_test, solver = 'sag', max_iter = 100):
    log =  LogisticRegression(solver=solver, max_iter = max_iter).fit(X_train, y_train)
    y_prob = log.predict_proba(X_test)
    y_prob_train = log.predict_proba(X_train)
    print ('Solver: {}, Max_Iter: {}' .format(solver, max_iter))
    printm(y_train, y_prob_train, y_test, y_prob)

def model_xgboost(X_train, X_test, y_train, y_test, eta = 0.3, gamma=0.1, max_depth=3, min_child_weight=3):
    xgb1 =  XGBClassifier(max_depth=max_depth, eta=eta, min_child_weight=min_child_weight).fit(X_train, y_train)
    y_prob = xgb1.predict_proba(X_test)
    y_prob_train = xgb1.predict_proba(X_train)
    print ('Max_depth: {}, min_child_weight: {}, eta: {}, gamma={}' .format(max_depth, min_child_weight, eta, gamma))
    printm(y_train, y_prob_train, y_test, y_prob)

def model_knn(X_train, X_test, y_train, y_test, n_neighbors=5, weights='uniform'):
    knn =  KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights).fit(X_train, y_train)
    y_prob = knn.predict_proba(X_test)
    y_prob_train = knn.predict_proba(X_train)
    print ('n_neighbors: {}, weights: {}' .format(n_neighbors, weights))
    printm(y_train, y_prob_train, y_test, y_prob)

def predict_test(model_selected, dev_size=0.10, oversampling=0):
    datax = read_datasets()
    datax1 = prep_variables(datax)

    X_train, X_test, y_train, y_test = train_test(datax1, test_size = dev_size, oversampling=oversampling)
    models = model_selected.fit(X_train, y_train)
    y_prob = models.predict_proba(X_test)
    y_prob_train = models.predict_proba(X_train)
    printm(y_train, y_prob_train, y_test, y_prob)

    test_df = read_datasets(list_data = ['final_txn_test_v2', 'final_user_log_test'])
    test_final = prep_variables(test_df)
    x_final, _, _, _ = train_test(test_final, test_size = 0.00, seed = 27, oversampling = 0)
    y_final = models.predict_proba(x_final)
    test_final['is_churn'] = y_final[:,1]

    new_id = pd.read_csv('../new_ids_v2.csv')
    #load new_id file
    test_final['new_id'] = test_final.index
    test_f = test_final[['new_id','is_churn']].merge(new_id, on='new_id', how='inner')
    test_f.drop('new_id', axis=1, inplace=True)
    test_f.to_csv('../test_prediction.csv', index=False)


if __name__ == '__main__':
    #datax = read_datasets()
    #datax1 = prep_variables(datax)

    #datax1['last_cancel'] = datax1['last_cancel'].apply(pd.to_numeric, errors='coerce').fillna(0)

    predict_test(RandomForestClassifier(n_estimators= 25, criterion = 'entropy', max_depth=9))

    #X_train, X_test, y_train, y_test = train_test(datax1, oversampling=1)
    #model_forest(X_train, X_test, y_train, y_test, trees=25, criterion = 'gini', max_depth=10)
    #model_forest(X_train, X_test, y_train, y_test, trees=15, criterion = 'gini', max_depth=10)
    #model_forest(X_train, X_test, y_train, y_test, trees=15, criterion = 'entropy', max_depth=10)
    #model_forest(X_train, X_test, y_train, y_test, trees=25, criterion = 'gini', max_depth=15)

    #X_train, X_test, y_train, y_test = train_test(datax1, oversampling=0)
    #model_forest(X_train, X_test, y_train, y_test, trees=25, criterion = 'gini', max_depth=10)

    #model_svm(X_train, X_test, y_train, y_test)
    #model_log(X_train, X_test, y_train, y_test)
    #model_gaussian(X_train, X_test, y_train, y_test)
    #model_knn(X_train, X_test, y_train, y_test)
    #model_xgboost(X_train, X_test, y_train, y_test)
