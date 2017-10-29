#!/usr/bin/env python

import pandas as pd
import numpy as np
import os.path
from os import listdir
import subprocess
import time

def split_user_logs_new(infilepath='../user_log_files/user_logs.csv', chunksize = 15000000):
    if os.path.isdir('../user_log_files'):
        if os.path.isfile('../user_log_files/user_logs0.csv'):
            print ('Files for user_logs.csv already existed. Nothing created')
            return False
        else:
            fname, ext = infilepath.rsplit('.',1)
            i = 0
            written = False
            with open(infilepath) as infile:
                while True:
                    outfilepath = "{}{}.{}".format(fname, i, ext)
                    with open(outfilepath, 'w') as outfile:
                        for line in (infile.readline() for _ in range(chunksize)):
                            outfile.write(line)
                        written = bool(line)
                    if not written:
                        break
                    i += 1

            print ('%s files generated' %(i))
            return True
    else:
        print ('Please create a folder "user_log_files and place user_logs.csv in that file"')
        return False


##---Create File with new Ids.
##---Should we consider onnly user ids that are presentin train and test set??
def create_new_ids(force = 0):
    """
    Create a new file with numerical ids for all all users
    """
    if os.path.isfile('../new_ids.csv') and force == 0:
        print ("File new_ids.csv already existed. Nothing created")
        return False
    else:
        train = pd.read_csv('../train.csv')
        #transactions = pd.read_csv('../transactions.csv')
        test = pd.read_csv('../sample_submission_zero.csv')

        #train_txn_temp = train.append(transactions, ignore_index = True)
        train_txn = train.append(test, ignore_index = True)

        #recreate ids (to reduce memory consumption by the long strings for both train and test msno's)
        ids = pd.DataFrame(np.unique(train_txn['msno']), columns = ['msno'])
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
def save_files_new_ids(files = ['members.csv', 'train.csv', 'transactions.csv', 'sample_submission_zero.csv'], prefix = 'new_', force = 0, force_userlog = 0):
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
            if doc != '.DS_Store' and doc != 'user_logs.csv' and doc[0:len(prefix)] != prefix:
                print ('updating ids for file %s ......'  %(doc))
                if doc == 'user_logs0.csv':
                    filex = pd.read_csv('../user_log_files/'+doc)
                else:
                    filex = pd.read_csv('../user_log_files/'+doc, header=None,
                        names = [
                            'msno', 'date', 'num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs'
                            ])
                filex = filex.merge(new_ids, left_on='msno', right_on ='msno', how='inner', copy = False).drop('msno', axis = 1)
                filex.to_csv('../user_log_files/'+prefix+doc, index = False)

    return True

################################################################################
#   Functions to read the files with the correct data types
################################################################################
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
                    'new_id': np.uint32,
                    'date': 'str',
                    'num_25': np.uint16,
                    'num_50': np.uint16,
                    'num_75': np.uint16,
                    'num_985': np.uint16,
                    'num_100': np.uint16,
                    'num_unq': np.uint16,
                    'total_secs': np.float32
                }, parse_dates = ['date'])
            if counter == 0:
                user_log_combined = user_log_temp
            else:
                user_log_combined = user_log_combined.append(user_log_temp, ignore_index=True)
            counter =+ 1

    return user_log_combined


def date_matrices(verbose = 1):

    [ print ('\nReading Datasets...\n') if verbose == 1 else 0]
    [ print ('\nReading Train...\n') if verbose == 1 else 0]
    tn = train()
    [ print ('\nReading Members...\n') if verbose == 1 else 0]
    memb = members()
    [ print ('\nReading Transactions...\n') if verbose == 1 else 0]
    txn = transactions()
    [ print ('\nReading User Logs...\n') if verbose == 1 else 0]
    ul = user_logs()
    [ print ('\nDatasets Loaded\n') if verbose == 1 else 0]

    ##Members DataFrame
    #(1) - add a lifetime(days) fiel from registration to expiration
    [ print ('Members: adding lifetime count (expiration - registration)\n') if verbose == 1 else 0]
    memb['lifetime'] = memb['expiration_date'] - memb['registration_init_time']
    memb['lifetime'] = memb['lifetime'].apply(lambda x: x.days)

    ##Transactions DataFrame
    #(1)-

    ## User Log DataFrame

    [ print ('User Log: Creating total_songs field\n') if verbose == 1 else 0]
    ul['total_songs']=(0.25*ul['num_25'] + 0.50*ul['num_50'] + 0.75*ul['num_75'] + 0.985*ul['num_985']+ul['num_100'])
    ul['total_songs'] = ul['total_songs'].clip(lower = 0, upper = ul['total_songs'].quantile(0.995))

    print (np.min(ul['total_songs']))
    [ print ('quantile 0.025 total_songs uncapped: {0} songs\n' .format(ul['total_songs'].quantile(0.025))) if verbose == 1 else 0]
    [ print ('quantile 0.500 total_songs uncapped: {0} songs\n' .format(ul['total_songs'].quantile(0.50))) if verbose == 1 else 0]
    [ print ('quantile 0.985 total_songs uncapped: {0} songs\n' .format(ul['total_songs'].quantile(0.985))) if verbose == 1 else 0]
    [ print ('quantile 0.995 total_songs uncapped: {0} songs\n' .format(ul['total_songs'].quantile(0.995))) if verbose == 1 else 0]

    [ print ('User Log: Calculate median song length to replace outliers\n') if verbose == 1 else 0]
    ul['avg_song'] = ul['total_secs']/(((0.25*ul['num_25'] + 0.50*ul['num_50'] + 0.75*ul['num_75'] + 0.985*ul['num_985'])*0.90)+ul['num_100'])
    avg_song = ul['avg_song'].quantile(0.5) #median

    [ print  ('quantile 0.025 avg_song uncapped: {0}s\n' .format(ul['avg_song'].quantile(0.025))) if verbose == 1 else 0]
    [ print  ('quantile 0.500 avg_song uncapped: {0}s\n' .format(ul['avg_song'].quantile(0.50))) if verbose == 1 else 0]
    [ print  ('quantile 0.985 avg_song uncapped: {0}s\n' .format(ul['avg_song'].quantile(0.985))) if verbose == 1 else 0]

    [ print ('User Log: Replace lower and upper outliers (median song length * songs played)\n') if verbose == 1 else 0]
    #ul[ul['total_secs']<=0]['total_secs'] = ul[ul['total_secs']<=0]['total_songs']*avg_song
    #ul[ul['total_secs']>ul['total_songs'].quantile(0.985)]['total_secs'] = ul[ul['total_secs']>ul['total_songs'].quantile(0.985)]['total_songs']*avg_song

    mask_lower = ul.total_secs <= 0
    mask_upper = ul.total_secs > ul['total_songs'].quantile(0.985)
    column_name = 'total_secs'
    ul.loc[mask_lower, column_name] = ul.loc[mask_lower, 'total_songs']*avg_song
    ul.loc[mask_upper, column_name] = ul.loc[mask_upper, 'total_songs']*avg_song

    [ print ('User Log: Creating yearMonth field\n') if verbose == 1 else 0]
    ul = ul[['date', 'new_id', 'total_secs', 'total_songs']]
    ul['yearMonth'] = ul['date'].map(lambda x: 1000*x.year + x.month)

    [ print ('User Log: Pivoting Table >>>Sum Secs<<<\n') if verbose == 1 else 0]
    ul_tsecs_month = pd.pivot_table(ul, values='total_secs', index=['new_id'], columns=['yearMonth'], aggfunc=np.sum)
    [ print ('User Log: Pivoting Table >>>Avg Secs<<<\n') if verbose == 1 else 0]
    ul_tsecs_month_mean = pd.pivot_table(ul, values='total_secs', index=['new_id'], columns=['yearMonth'], aggfunc=np.mean)
    [ print ('User Log: Pivoting Table >>>Sum Songs<<<\n') if verbose == 1 else 0]
    ul_tsongs_month = pd.pivot_table(ul, values='total_songs', index=['new_id'], columns=['yearMonth'], aggfunc=np.sum)
    [ print ('User Log: Pivoting Table >>>Avg Songs<<<\n') if verbose == 1 else 0]
    ul_tsongs_month_mean = pd.pivot_table(ul, values='total_songs', index=['new_id'], columns=['yearMonth'], aggfunc=np.mean)

    [ print ('User Log: Saving Tables\n') if verbose == 1 else 0]
    ul_tsecs_month.to_csv('../ul_tsecs_month.csv')
    ul_tsecs_month_mean.to_csv('../ul_tsecs_month_mean.csv')
    ul_tsongs_month.to_csv('../ul_tsongs_month.csv')
    ul_tsongs_month_mean.to_csv('../ul_tsongs_month_mean.csv')

    return True

if __name__ == '__main__':
    #(0)split user_log file
    #split_user_logs_new()

    #(1)create file new_ids if it does not exist
    #create_new_ids(force = 0)

    #(2)change ids in the main files (except user_logs), if it does not exist yet
    #save_files_new_ids(force = 0, force_userlog = 0)
    date_matrices(verbose=1)
