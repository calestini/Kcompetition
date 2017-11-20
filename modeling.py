#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
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
from imblearn.over_sampling import SMOTE


def read_datasets(list_data = ['final_txn_v2', 'final_user_log_v2']):
    for i in range(0, len(list_data)):
        temp_df = pd.read_csv('../'+list_data[i]+'.csv')
        if i == 0:
            merged_df = temp_df
        else:
            merged_df = merged_df.merge(temp_df, on='new_id', how='inner')

    return merged_df

def prep_variables():
    train = read_datasets()

    #DUMMIES FOR MONTHS_LISTENING
    train = train.join(pd.get_dummies(train['months_listening'])).drop('months_listening', axis = 1)

    #DUMMIES FOR CHURN
    train = train.join(pd.get_dummies(train['cluster9'])).drop('cluster9', axis = 1)

    return train

def oversample(train):
    y = train['is_churn']
    x = train.drop('is_churn', axis = 1)

    sm = SMOTE(random_state=27) #check for parameters tunning
    x_res, y_res = sm.fit_sample(x, y)
    return x_res, y_res

def train_test(test_size = 0.20, seed = 27, oversample = 1):
    data_model = {}

    train = prep_variables()
    if oversample == 1:
        x, y = oversample(train)
    else:
        y = train['is_churn']
        x = train.drop('is_churn', axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = seed)

    data_model['xtrain'] = X_train
    data_model['xtest'] = X_test
    data_model['ytrain'] = y_train
    data_model['ytest'] = y_test

    return data_model

models = {}
models['LR']= LogisticRegression()
models['LDA']= LinearDiscriminantAnalysis()
models['KNN']= KNeighborsClassifier()
models['DT']= DecisionTreeClassifier()
models['RF']= RandomForestClassifier()
models['NB']= GaussianNB()
models['SVM']= SVC()

def model(data_model, classifier='RF', parameters):
    '''
    generic function to accept any model and parameters for the model
    :data_model: [dictionary] data with train and test splits
    :classifier: [string] algorithm to use for classficiation
    :parameters: [dictionary]
    '''
    X_train = data_model['xtrain']
    X_test = data_model['xtest']
    y_train = data_model['ytrain']
    y_test = data_model['ytest']


def model_forest(data_model, logprint=1):
    '''
    random forest model
    :data_model: train and test[dev] splits
    :return:
    '''
    X_train = data_model['xtrain']
    X_test = data_model['xtest']
    y_train = data_model['ytrain']
    y_test = data_model['ytest']

    for depth in range(15,30,1):
        forest = RandomForestClassifier(n_estimators= 100, criterion = 'entropy', random_state = 0, max_depth=depth)
        forest.fit(X_train, y_train)
        y_pred = forest.predict(X_test)
        y_prob = forest.predict_proba(X_test)
        y_prob_train = forest.predict_proba(X_train)
        print ('Depth: \t\t {0}'.format(depth))
        print('Random Forest Train \t{:.4f}' .format(log_loss(y_train, y_prob_train)))
        print('Random Forest Dev \t{:.4f}' .format(log_loss(y_test, y_prob)))

def decision_tree(data_model, logprint=1):
    X_train = data_model['xtrain']
    X_test = data_model['xtest']
    y_train = data_model['ytrain']
    y_test = data_model['ytest']

    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, max_depth = 15)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    y_prob = tree.predict_proba(X_test)
    y_prob_train = tree.predict_proba(X_train)
    print('Decision Tree Train \t{:.4f}' .format(log_loss(y_train, y_prob_train)))
    print('Decision Tree Dev \t{:.4f}' .format(log_loss(y_test, y_prob)))

def modelling(logprint = 1):
    #STEP0: READ DATASETS
    train = read_datasets()

    #STEP1: OVERSAMPLE
    otrain = train#oversample(train)

    #STEP1.5: CHECK FOR NULLS
    #print message if there are nulls

    #STEP2: GET DUMMIES FOR CATEGORICAL
    #DUMMIES FOR MONTHS_LISTENING
    otrain = otrain.join(pd.get_dummies(otrain['months_listening'])).drop('months_listening', axis = 1)

    #DUMMIES FOR CHURN
    otrain = otrain.join(pd.get_dummies(otrain['cluster9'])).drop('cluster9', axis = 1)

    #STEP3: SPLIT TRAIN TEST
    y = otrain['is_churn']
    x = otrain.drop('is_churn', axis = 1)

    #STEP4: RUN DECISION TREE
    depthss = []
    loglosss = []
    for depth in range(7,8):
        depthss.append(depth)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1)

        # Fitting Decision Tree Classification to the Training set
        tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, max_depth = depth)
        tree.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = tree.predict(X_test)
        y_prob = tree.predict_proba(X_test)

        # Making the Confusion Matrix
        #cm = confusion_matrix(y_test, y_pred)

        #logloss
        print(log_loss(y_test, y_prob))

        loglosss.append(log_loss(y_test, y_prob))
        #print ('Error: {:0.2f}%' .format(((cm[0][1]+ cm[1][0])/(cm[1][1]+ cm[0][0]))*100))

    forest = RandomForestClassifier(n_estimators= 100, criterion = 'entropy', random_state = 0)
    #forest.fit(X_train, y_train)

    # Predicting the Test set results
    #y_pred = forest.predict(X_test)
    #y_prob = forest.predict_proba(X_test)
    #print ('Forest')
    #print(log_loss(y_test, y_prob))

    #print ('accuracies CV')
    #accuracies = cross_val_score(estimator = forest, X = X_train, y = y_train, cv = 10)
    #print (accuracies.mean())
    #print (accuracies.std())

    # Random Forest Grid Search
    parameters = [{"max_depth": [15, 25]}]

    #Grid search, optimizing for log loss
    grid_search = GridSearchCV(estimator = forest,
                               param_grid = parameters,
                               scoring = 'neg_log_loss',
                               cv = 5,
                               n_jobs = -1)
    #grid_search = grid_search.fit(X_train, y_train)

    #Grid search results
    #best_accuracy = grid_search.best_score_
    #best_parameters = grid_search.best_params_

    #print ('grid search forest')
    #print (best_accuracy)
    #print (best_parameters)

    from xgboost import XGBClassifier
    xg = XGBClassifier()
    xg.fit(X_train, y_train)
    y_prob = xg.predict_proba(X_test)

    print ('xgboost')
    print (log_loss(y_test, y_prob))

if __name__ == '__main__':
    #model()
    data_model = train_test()
    model_forest(data_model)
