#!/usr/bin/env python

import numpy as np
import pandas as pd
import modeling as mdl
import seaborn as sns
import matplotlib.pyplot as plt
import data_manipulation_v2 as dm
import os.path
from os import listdir

def create_add_new_ids(force = 0):
    """
    Create a new file only with numerical ids for all all users, instead of msno.
    This file can be then used to trace back

    Create new ids modified to take in new files
    """
    if os.path.isfile('../new_ids_add.csv') and force == 0:
        print ("File new_ids_add.csv already existed. Nothing created")
        return False
    else:
        #Load Files
        new_id = pd.read_csv('../new_ids.csv')
        new_id_v2 = pd.read_csv('../new_ids_v2.csv')
        train = dm.train()
        
        #Assign msnos back to train files
        train = train.merge(new_id, on = 'new_id', how = 'inner')
        feb_churners_append = train[train['msno'].isin(new_id_v2['msno'])==False]['msno'].reset_index(name='msno')
        
        #Determine min and max new ids
        min_id=max(new_id_v2['new_id'])+1
        max_id=max(new_id_v2['new_id'])+feb_churners_append['msno'].nunique()+1
                  
        feb_churners_append['new_id'] = range(min_id, max_id)

        feb_churners_append.to_csv('../new_ids_add.csv', index = False)
        print ('\n\tFile new_ids_add.csv created successfully!\n\t')

        return True
    

def save_files_add_ids(files = ['merged_transactions.csv', 'members.csv', 'train.csv'], prefix = 'add_', force = 0, force_userlog = 0):
    """
    Function to save .csv files with the new numerical id (as opposed to msno).

    :files: list of files to replace ids
    :predix: string to add to new files
    :force: whether to overwrite in case new files already exist. 0 won't overwrite
    :force_userlog: whether to overwirte but for user_log split files. 0 won't overwrite

    Updated to read in members_v2.csv, train_v2.csv, sample_submission_v2.csv, and merged transactions file.
    Updated to read in user_logs_v2.csv file with headers.
    """
    new_ids = pd.read_csv('../new_ids_add.csv')
    list_files = files

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
            if doc != '.DS_Store' and doc != 'user_logs.csv' and doc[0:len(prefix)] != prefix and doc[0:len(prefix)] != 'new_':
                print ('updating ids for file %s ......'  %(doc))
                if (doc == 'user_logs0.csv')|(doc == 'user_logs_v2.csv'):
                    filex = pd.read_csv('../user_log_files/'+doc)
                else:
                    filex = pd.read_csv('../user_log_files/'+doc, header=None,
                        names = [
                            'msno', 'date', 'num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs'
                            ])
                filex = filex.merge(new_ids, left_on='msno', right_on ='msno', how='inner', copy = False).drop('msno', axis = 1)
                filex.to_csv('../user_log_files/'+prefix+doc, index = False)

    return True


if __name__ == '__main__':
    
    #create_add_new_ids()
    save_files_add_ids(force_userlog=1)



