#!/usr/bin/env python
import numpy as np
import pandas as pd
import data_manipulation_v2 as dm
import datetime as dt
import functools

def txn_train_test(dataset):
    
    print ('Reading datasets...')
    train = dm.train_v2()
    test = dm.test_v2()
    txn_og = dm.transactions_merged() 
    memb = dm.members_v2()
    
    print ('Sorting transactions by new_id and dates...')
    txn_og = txn_og.sort_values(['new_id', 'transaction_date', 'membership_expire_date'])
    
    if dataset == 'train':
        print ('Filtering transaction dataset by transaction date for train set...')
        txn=txn_og[txn_og['transaction_date']<='2017-02-28']
    else: txn=txn_og

    #Number of transactions
    txn_cnts = txn.groupby(['new_id']).size().reset_index(name='txn_cnt')
    
    #Total payment plan days, plan list price, actual amount paid
    tot_plan_pmt = txn.groupby(['new_id'])[['payment_plan_days', 'plan_list_price', 'actual_amount_paid']].sum().reset_index()
    tot_plan_pmt['avg_daily_paid'] = tot_plan_pmt['actual_amount_paid'] / tot_plan_pmt['payment_plan_days']
    tot_plan_pmt['list_actual_diff'] = tot_plan_pmt['plan_list_price'] - tot_plan_pmt['actual_amount_paid']
    
    #Number, and Percentage of Transactions where Plan List Price Higher than Actual Amount Paid
    txn['list_actual_diff'] = txn['plan_list_price'] - txn['actual_amount_paid']
    txn_lp_high = txn[txn['list_actual_diff'] > 0].groupby('new_id')['list_actual_diff'].count().reset_index(name='lp_high_cnt')
    txn_lp_high = txn_cnts.merge(txn_lp_high, on = 'new_id', how = 'left')
    txn_lp_high['per_lp_high'] = txn_lp_high['lp_high_cnt']/txn_lp_high['txn_cnt']
    txn_lp_high.drop('lp_high_cnt', axis = 1, inplace = True)
    
    #Previous churn: determine if transaction date-membership expiry date from previous row is >30days 
    txn['shifted_expiry'] = txn.groupby('new_id')['membership_expire_date'].shift()
    txn['memb_gap'] = (txn['transaction_date'] - txn['shifted_expiry'])
    txn['prev_churn'] = (txn['memb_gap']/np.timedelta64(1, 'D'))>30
    #Aggregate number of times churned and calculate number of times churned as percentage of total transactions
    txn_prev_churn = txn[txn['prev_churn'] == True].groupby('new_id')['prev_churn'].count().reset_index(name='prev_churn_cnt')
    txn_prev_churn = txn_prev_churn.merge(txn_cnts, on='new_id', how='inner')
    txn_prev_churn['prev_churn_per'] = txn_prev_churn['prev_churn_cnt'] / txn_prev_churn['txn_cnt']
    txn_prev_churn.drop(['prev_churn_cnt', 'txn_cnt'], axis = 1, inplace = True)
    
    #Value of 'is_auto_renew' for last transaction
    txn_ar_last = txn.groupby(['new_id'])['is_auto_renew'].last().reset_index(name='last_ar')
    #Changed auto renew:
    txn_ar_change = txn.groupby(['new_id'])['is_auto_renew'].mean().reset_index(name='mean_ar')
    txn_ar_stop = txn_ar_change.merge(txn_ar_last, on='new_id', how='inner')
    txn_ar_stop['stopped_ar']=(txn_ar_stop['mean_ar'] < 1) & (txn_ar_stop['mean_ar'] > 0) & (txn_ar_stop['last_ar'] == 0)
    txn_ar_stop.drop('last_ar', axis = 1, inplace = True)

    #Changed payment method:determine if transaction date-membership expiry date from previous row is >30days 
    txn['shifted_pmt'] = txn.groupby('new_id')['payment_method_id'].shift()
    txn['pmt_change'] = (txn['payment_method_id'] != txn['shifted_pmt']) & (txn['shifted_pmt'].isnull() != True)
    #Aggregate number of times changed payment method
    txn_pmt_change = txn[txn['pmt_change'] == True].groupby('new_id')['pmt_change'].count().reset_index(name='pmt_change_cnt')

    #Cancelled membership:
    txn_cancelled = txn.groupby(['new_id'])['is_cancel'].mean().reset_index(name='mean_cancel')
    #Cancelled membership in last transaction:
    txn_cancelled_last = txn.groupby(['new_id'])['is_cancel'].last().reset_index(name='last_cancel')
    
    #Free trials:
    #5483 members who have their last transaction as a free trial
    free_trial = txn.groupby(['new_id'])['plan_list_price'].last().reset_index(name='last_plan_price')
    free_trial['lst_free_trial'] = (free_trial['last_plan_price'] == 0) | (free_trial['last_plan_price'] == 1)
        
    #percentage of total transactions that are free trials
    free_trial_cnt = txn[(txn['plan_list_price'] == 0) | (txn['plan_list_price'] == 1)].groupby('new_id')['plan_list_price'].count().reset_index(name='free_trial_cnt')
    per_free_trial = txn_cnts.merge(free_trial_cnt, on = 'new_id', how = 'left')
    per_free_trial['per_free_trial'] = per_free_trial['free_trial_cnt']/per_free_trial['txn_cnt']
    free_trial = free_trial.merge(per_free_trial, on = 'new_id', how = 'inner')
        
    #percentage of total payment_plan_days that are free trial days
    free_trial_days = txn[(txn['plan_list_price'] == 0) | (txn['plan_list_price'] == 1)].groupby('new_id')['payment_plan_days'].sum().reset_index(name='free_trial_days')
    per_days_free_trial = tot_plan_pmt[['payment_plan_days', 'new_id']].merge(free_trial_days, on = 'new_id', how = 'left')
    per_days_free_trial['per_days_free_trial'] = per_days_free_trial['free_trial_days']/per_days_free_trial['payment_plan_days']
    free_trial = free_trial.merge(per_days_free_trial, on = 'new_id', how = 'inner')
    free_trial.drop(['last_plan_price', 'txn_cnt', 'free_trial_cnt', 'payment_plan_days', 'free_trial_days'], axis = 1, inplace = True)

    #max_memb_expire and lst_memb_expire
    max_memb_expire = txn.groupby(['new_id'])['membership_expire_date'].max().reset_index(name='max_memb_expire')
    lst_memb_expire = txn.groupby(['new_id'])['membership_expire_date'].last().reset_index(name='lst_memb_expire')
    fst_txn_dt = txn.groupby(['new_id'])['transaction_date'].first().reset_index(name='fst_transaction_date')
    memb_expire = max_memb_expire.merge(lst_memb_expire, on = 'new_id', how = 'inner')
    memb_expire = memb_expire.merge(fst_txn_dt, on = 'new_id', how = 'left')
    memb_expire['not_equal'] = memb_expire['max_memb_expire'] != memb_expire['lst_memb_expire']

    if dataset == 'train':
        memb_expire['lst_memb_expire_post'] = memb_expire['lst_memb_expire'] >= '2017-04-01'
    else:
        memb_expire['lst_memb_expire_post'] = memb_expire['lst_memb_expire'] >= '2017-05-01'
        
    memb_expire = memb_expire.merge(memb[['new_id', 'registration_init_time']], on = 'new_id', how = 'left')
    memb_expire['registration_init_time'] = memb_expire['registration_init_time'].fillna(memb_expire['fst_transaction_date'])
    
    if dataset == 'train':
        memb_expire['memb_tenure_days'] = (pd.to_datetime('2017-02-28') - memb_expire['registration_init_time'])/np.timedelta64(1, 'D')
    else:
        memb_expire['memb_tenure_days'] = (pd.to_datetime('2017-03-31') - memb_expire['registration_init_time'])/np.timedelta64(1, 'D')
    
    memb_expire['lst_memb_expire_days'] = (memb_expire['lst_memb_expire'] - pd.to_datetime('2017-03-31'))/np.timedelta64(1, 'D')
    memb_expire.drop(['max_memb_expire', 'registration_init_time', 'lst_memb_expire', 'fst_transaction_date'], inplace = True, axis = 1)

    if dataset == 'train':
        print('Merging transaction features with train dataset...')
        txn_features = [
            train, tot_plan_pmt, txn_ar_stop, txn_cancelled, 
            txn_cancelled_last, free_trial, txn_lp_high, 
            txn_prev_churn, txn_pmt_change, memb_expire]
        f_txn = functools.reduce(lambda left,right: pd.merge(left,right,on='new_id', how='left'), txn_features)
    else:
        print('Merging transaction features with test dataset...')
        txn_features = [
            test, tot_plan_pmt, txn_ar_stop, txn_cancelled, 
            txn_cancelled_last, free_trial, txn_lp_high, 
            txn_prev_churn, txn_pmt_change, memb_expire]
        f_txn = functools.reduce(lambda left,right: pd.merge(left,right,on='new_id', how='left'), txn_features)
    
    print ('Replacing null values...')
    str_col = ['per_free_trial', 'per_days_free_trial', 'txn_cnt'
               , 'per_lp_high', 'prev_churn_per', 'pmt_change_cnt', 'lst_memb_expire_days', 'memb_tenure_days']
    f_txn[str_col] = f_txn[str_col].fillna(0)
    
    str_col = ['stopped_ar', 'last_cancel', 'lst_free_trial', 'not_equal', 'lst_memb_expire_post']
    f_txn[str_col] = f_txn[str_col].fillna(False)
    
    str_col = ['payment_plan_days', 'plan_list_price', 'actual_amount_paid']
    f_txn[str_col] = f_txn[str_col].fillna(f_txn[str_col].mode().iloc[0])
    
    str_col = ['mean_ar', 'mean_cancel']
    f_txn[str_col] = f_txn[str_col].fillna(f_txn[str_col].mean().iloc[0])
    
    f_txn['avg_daily_paid'] = f_txn['actual_amount_paid']/f_txn['payment_plan_days']
    f_txn['list_actual_diff'] = f_txn['plan_list_price'] - f_txn['actual_amount_paid']

    if f_txn.isnull().sum().sum()>0:
        print('Something is wrong! There are null values in the dataframe')
        print(f_txn.info())
     
    #Drop 'is_churn' field:
    f_txn.drop('is_churn', axis = 1, inplace = True)
    
    #Export into csv:
    if dataset == 'train':
        print ('Exporting transaction features for train dataset into csv...') 
        f_txn.to_csv('../final_txn_v2.csv', index=False)
    else:
        print ('Exporting transaction features for test dataset into csv...') 
        f_txn.to_csv('../final_txn_test_v2.csv', index=False)
        
    return f_txn

txn_train_test(dataset='test')