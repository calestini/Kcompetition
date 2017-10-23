#!/usr/bin/env python

import pandas as pd
import numpy as np
import time
from datetime import datetime, date, timedelta

def prepare_transaction(filename = 'transactions.csv'):
    #############################################################################
    ###     STEP 1: READ DATASET
    #############################################################################
    print ('\n\treading dataset...')

    tic = time.time()

    transactions = pd.read_csv('../'+'transactions.csv')

    #############################################################################
    ###     STEP 2: FILTER / MODIFY TRANSACTION DATASET
    #############################################################################

    #keep only relevant columns
    transactions1 = transactions[['msno','transaction_date','membership_expire_date']]
    del transactions

    #transform into datetime
    transactions1['transaction_date'] = pd.to_datetime(transactions1['transaction_date'].map(str), format='%Y%m%d')
    transactions1['membership_expire_date'] = pd.to_datetime(transactions1['membership_expire_date'].map(str), format='%Y%m%d')

    #transform into date only
    transactions1['transaction_date'].apply(lambda x: x.date())
    transactions1['membership_expire_date'].apply(lambda x: x.date())

    toc = time.time()
    print ('...dataset ready - %s ms\n...' %(toc-tic))

    return transactions1

#find a way to create a chrnonology of users behaviors
def txn_days(follow_days = 790):
    #############################################################################
    ###     STEP 1: CREATE LIST OF DATES
    #############################################################################

    #starting on 2015/01/01 until 2017/02/28 (transactions range)
    transaction_days = []
    for i in range(follow_days):
        transaction_days.append([date(year=2015,day=1,month=1)+timedelta(days=i)])

    return np.asarray(transaction_days)

if __name__ == '__main__':
    txn = prepare_transaction()
    days = txn_days()
    #print (txn['transaction_date'] - date(year=2015,day=1,month=1))
