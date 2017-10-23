#!/usr/bin/env python
import pandas as pd
import numpy as np
import os.path
import subprocess
import time

##---Create File with new Ids.
def create_new_ids():
    """
    Create a new file with numerical ids for all all users
    """
    if os.path.isfile('../new_ids.csv'):
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

        ids.to_csv('../new_ids.csv')
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

if __name__ == '__main__':
    #(1)create file new_ids if it does not exist
    create_new_ids()

    #(2)split user_log.csv into many files of 1,000 MB
    split_user_logs()
