#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

def read_datasets(list_data = ['final_txn_v2', 'final_user_log_v2']):
    for i in range(0, len(list_data)):
        temp_df = pd.read_csv('../'+list_data[i]+'.csv')
        if i == 0:
            merged_df = temp_df
        else:
            merged_df = merged_df.merge(temp_df, on='new_id', how='inner')

    return merged_df


def oversample(train):
    tcols = train.columns
    train_np = train.values
    train_yes = train_np[train_np[:,0] == 1, :]
    train_no = train_np[train_np[:,0] == 0, :]

    repeat_number = int(train_no.shape[0]/train_yes.shape[0])

    ntrain_yes = train_yes.repeat(repeat_number, axis = 0)
    ntrain_list = train_yes.tolist() + train_no.tolist()

    [print ('Error!!! did not merge properly') if len(ntrain_list) != (train_yes.shape[0] + train_no.shape[0]) else False]
    ntrain = pd.DataFrame(ntrain_list, columns = tcols)
    return ntrain

def model(logprint = 1):
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
    for depth in range(7,11):
        depthss.append(depth)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1)

        y_train = y_train.values.reshape(len(y_train), 1)
        y_test = y_test.values.reshape(len(y_test), 1)

        #print(y_train.shape)
        #print(y_test.shape)
        #print(X_train.shape)
        #print(X_test.shape)

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

    forest = RandomForestClassifier(n_estimators = 10, random_state = 0)
    forest.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = forest.predict(X_test)
    y_prob = forest.predict_proba(X_test)
    print ('Forest')
    print(log_loss(y_test, y_prob))

    #print ('accuracies CV')
    #accuracies = cross_val_score(estimator = forest, X = X_train, y = y_train, cv = 10)
    #print (accuracies.mean())
    #print (accuracies.std())

    # Random Forest Grid Search
    parameters = [{"max_depth": [3, None],
                  "max_features": [1, 3, 10],
                  "min_samples_split": [2, 3, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}]

    #Grid search, optimizing for log loss
    grid_search = GridSearchCV(estimator = forest,
                               param_grid = parameters,
                               scoring = 'neg_log_loss',
                               cv = 10,
                               n_jobs = -1)
    grid_search = grid_search.fit(X_train, y_train)

    #Grid search results
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_

    print ('grid search forest')
    print (best_accuracy)

if __name__ == '__main__':
    model()
