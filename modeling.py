#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, log_loss

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
    for depth in range(7,27):
        depthss.append(depth)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 1)

        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)

        #print(y_train.shape)
        #print(y_test.shape)
        #print(X_train.shape)
        #print(X_test.shape)

        # Fitting Decision Tree Classification to the Training set
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, max_depth = depth)
        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)

        # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        #logloss
        print(log_loss(y_test, y_pred))
        loglosss.append(log_loss(y_test, y_pred))
        #print ('Error: {:0.2f}%' .format(((cm[0][1]+ cm[1][0])/(cm[1][1]+ cm[0][0]))*100))

if __name__ == '__main__':
    model()
