#!/usr/bin/env python

import pandas as pd
import numpy as np
import os.path
from os import listdir
import subprocess
import time

##---Create File with new Ids.
def create_new_ids(force = 0):
    """
    Create a new file with numerical ids for all all users
    """
    if os.path.isfile('../new_ids.csv') and force == 0:
        print ("File new_ids.csv already existed. Nothing created")
        return False
    else:
        train = pd.read_csv('../train.csv')
        test = pd.read_csv('../test.csv')

        #concatenate both to get all msno ids
        test_train = pd.concat([train, test[['msno']] ])

        #recreate ids (to reduce memory consumption by the long strings for both train and test msno's)
        ids = pd.DataFrame(np.unique(test_train['msno']), columns = ['msno'])
        ids['new_id'] = range(len(ids))

        ids.to_csv('../new_ids.csv', index = False)
        print ('\n\tFile new_ids.csv created successfully!\n\t')

        return True

def split_user_logs():
    """
    Split user_logs.csv into multiple files
    """
    if os.path.isdir('../user_log_files'):
        if os.path.isfile('../user_log_files/xaa'):
            print ('Files for user_logs.csv already existed. Nothing created')
            return False
        else:
            bashCommand = "split -b 1000m user_logs.csv"

            print ('\n\tCreating split files. This may take some time...\n\t')
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, cwd='../user_log_files')
            output, error = process.communicate()

            return True
    else:
        print ('Please create a folder "user_log_files and place user_logs.csv in that file"')
        return False

#not for user logs yet
def save_files_new_ids(files = ['members.csv', 'train.csv', 'transactions.csv','test.csv'], prefix = 'new_', force = 0, force_userlog = 0):
    new_ids = pd.read_csv('../new_ids.csv')
    list_files = files

    #regular files
    for doc in list_files:
        if os.path.isfile('../' + prefix + doc) and force == 0:
            print ('File %s already existed. Nothing created' %(prefix+doc))
            pass
        else:
            print ('updating ids for file %s ......'  %(doc))
            filex = pd.read_csv('../'+doc)
            filex = filex.merge(new_ids, left_on='msno', right_on ='msno', how='inner', copy = False).drop('msno', axis = 1)
            filex.to_csv('../'+prefix+doc, index = False)

    #user_log split files
    if force_userlog != 0:
        onlyfiles = [f for f in listdir('../user_log_files/') if os.path.isfile(os.path.join('../user_log_files/', f))]

        for doc in onlyfiles:
            if doc != '.DS_Store' and doc != 'user_logs.csv':
                print ('updating ids for file %s ......'  %(doc))
                if doc == 'xaa':
                    filex = pd.read_csv('../user_log_files/'+doc)
                else:
                    filex = pd.read_csv('../user_log_files/xac', header=None,
                        names = [
                            'msno', 'date', 'num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs'
                            ])
                filex = filex.merge(new_ids, left_on='msno', right_on ='msno', how='inner', copy = False).drop('msno', axis = 1)
                filex.to_csv('../user_log_files/'+prefix+doc, index = False)

    return True

def members(new_ids = 'yes'):
    """
    Read and parse the info for the members file (smallint, dates, ...)
    """
    members_file = pd.read_csv('../new_members.csv', dtype={
            'bd': np.int8, 'city': np.int8, 'gender':'category' ,
            'registered_via':np.int8, 'registration_init_time':'str',
            'expiration_date': 'str', 'new_id':np.uint32},
                parse_dates=['expiration_date', 'registration_init_time'])

    return members_file

def transactions(new_ids = 'yes'):
    """
    Read and parse the info for the txn file (smallint, dates, ...)
    """
    txn_file = pd.read_csv('../new_transactions.csv', dtype={
        'payment_method_id': np.int8,
        'payment_plan_days': np.uint16,
        'is_cancel': np.int8,
        'is_auto_renew': np.int8,
        'transaction_date': 'str',
        'membership_expire_date': 'str',
        'plan_list_price': np.uint16,
        'actual_amount_paid': np.uint16,
        'new_id': np.uint32},
        parse_dates=['transaction_date', 'membership_expire_date'])

    return txn_file

def train(new_ids = 'yes'):
    """
    Read and parse the info for the train file (smallint, dates, ...)
    """
    train_file = pd.read_csv('../new_train.csv', dtype={
            'is_churn': np.int8,
            'new_id': np.uint16})

    return train_file

def user_logs(new_ids = 'yes', prefix = "new_"):
    """
    Read and parse the info for the user_log files (smallint, dates, ...)
    """
    onlyfiles = [f for f in listdir('../user_log_files/') if os.path.isfile(os.path.join('../user_log_files/', f))]
    counter = 0
    for doc in onlyfiles:
        if doc[0:len(prefix)] == prefix:
            user_log_temp = pd.read_csv('../user_log_files/'+doc, dtype = {
                    'msno': 'str',
                    'date': 'str',
                    'num_25': np.float16,
                    'num_50': np.float16,
                    'num_75': np.float16,
                    'num_985': np.float16,
                    'num_100': np.float16,
                    'num_unq': np.float16,
                    'total_secs': np.float16
                }, parse_dates = ['date'])
            if counter == 0:
                user_log_combined = user_log_temp
            else:
                user_log_combined = user_log_combined.append(user_log_temp, ignore_index=True)

    return user_log_combined


if __name__ == '__main__':
    #(1)create file new_ids if it does not exist
    create_new_ids(force = 0)

    #(2)split user_log.csv into many files of 1,000 MB
    split_user_logs()

    #(3)change ids in the main files (except user_logs), if it does not exist yet
    save_files_new_ids(force_userlog = 0)

    tic = time.time()
    x1 = members()
    x2 = transactions()
    x3 = train()
    #x4 = user_logs() #this might take a while
    toc = time.time()

    print('it took %s ms' %(toc-tic))
