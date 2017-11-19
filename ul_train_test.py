#!/usr/bin/env python
import numpy as np
import pandas as pd
import data_manipulation_v2 as dm
import datetime as dt

print ('Reading datasets...')
train = dm.train_v2()
ul = dm.user_logs() #it already uses v2 train and test ids
test = dm.test_v2()

print ('Sorting user log by new_id and date...')
ul = ul.sort_values(['new_id','date'])

#CREATE FREQ_DAYS VARIABLE
ul['previous_date'] = ul.groupby('new_id')['date'].shift()
ul['time_lapsed'] = (ul['date'] - ul['previous_date']).astype('timedelta64[D]')
ul_freq = ul.groupby('new_id')['time_lapsed'].median().reset_index(name='freq_days')

#ESTIMATE total number of songs and recalculate total_secs for outliers
ul['total_songs']=(0.25*ul['num_25'] + 0.50*ul['num_50'] + 0.75*ul['num_75'] + 0.985*ul['num_985']+ul['num_100'])
ul['total_songs'] = ul['total_songs'].clip(lower = 0, upper = ul['total_songs'].quantile(0.995))

compensation_factor = 1.0

ul['avg_song'] = ul['total_secs']/(((0.25*ul['num_25'] + 0.50*ul['num_50'] + 0.75*ul['num_75'] + 0.985*ul['num_985'])*compensation_factor)+ul['num_100'])
avg_song = ul['avg_song'].quantile(0.5) #median

mask_lower = ul.total_secs <= 0
mask_upper = ul.total_secs > ul['total_secs'].quantile(0.985)
column_name = 'total_secs'
ul.loc[mask_lower, column_name] = ul.loc[mask_lower, 'total_songs']*avg_song
ul.loc[mask_upper, column_name] = ul.loc[mask_upper, 'total_songs']*avg_song

#CREATE STDEV_SECS VARIABLE
ul['logsecs'] = np.log(ul['total_secs'])
ul_csecs = ul.groupby('new_id')['logsecs'].std().reset_index(name='std_logsecs')

#ADD A YEARMONTH VARIABLE TO AGGREGATE BY MONTH
ul['yearMonth'] = pd.DatetimeIndex(ul['date']).year*100+pd.DatetimeIndex(ul['date']).month

#PIVOT
sum_months = pd.pivot_table(ul, values='total_secs', index=['new_id'], columns=['yearMonth'], aggfunc=np.sum)

#CREATE MONTHS LISTNEING VARIABLE
sum_months['months_listening'] = pd.cut(np.count_nonzero(sum_months.fillna(0).drop(201703,axis=1), axis = 1)
                                        , [0,6,12,18,24,30]
                                        , labels=['0-6','7-12','13-18','19-24','25-27']).fillna('0-6')

#CREATE LISTENING PREVIOUS 6 AND PREVIOUS 12
#Mar skipped for training set (roll forward for test set)

p6_train = [201702,201701,201612,201611,201610,201609]
p12_train = [201702,201701,201612,201611,201610,201609,201608,201607,201606,201605,201604,201603]
p6_test = [201703,201702,201701,201612,201611,201610]
p12_test = [201703,201702,201701,201612,201611,201610,201609,201608,201607,201606,201605,201604]

sum_months['listening_p6_train'] = np.count_nonzero(sum_months[p6_train].fillna(0), axis = 1)
sum_months['listening_p12_train'] = np.count_nonzero(sum_months[p12_train].fillna(0), axis = 1)
sum_months['listening_p6_test'] = np.count_nonzero(sum_months[p6_test].fillna(0), axis = 1)
sum_months['listening_p12_test'] = np.count_nonzero(sum_months[p12_test].fillna(0), axis = 1)

#CREATE LOG OF MONTHLY AVERAGE
sum_months['logavg_secs_p12_train'] = np.log((np.sum(sum_months[p12_train].fillna(0), axis = 1)+1)/(sum_months['listening_p12_train']+1))
sum_months['logavg_secs_p12_test'] = np.log((np.sum(sum_months[p12_test].fillna(0), axis = 1)+1)/(sum_months['listening_p12_test']+1))

##CREATE VAR OF PREVIOUS CONSECUTIVE MONTHS W/O LISTENING DATA
#Mar skipped for training set (roll forward for test set)
no_songs = []
for id in np.unique(sum_months.index.values):
    total_months = 0
    if np.isnan(sum_months.loc[id, 201702]):
        total_months += 1
        if np.isnan(sum_months.loc[id, 201701]):
            total_months += 1
            if np.isnan(sum_months.loc[id, 201612]):
                total_months += 1
                if np.isnan(sum_months.loc[id, 201611]):
                    total_months += 1
                    if np.isnan(sum_months.loc[id, 201610]):
                        total_months += 1
                        if np.isnan(sum_months.loc[id, 201609]):
                            total_months += 1
    no_songs.append(total_months)

sum_months['no_songs_cp6_train'] = no_songs

no_songs = []
for id in np.unique(sum_months.index.values):
    total_months = 0
    if np.isnan(sum_months.loc[id, 201703]):
        total_months += 1
        if np.isnan(sum_months.loc[id, 201702]):
            total_months += 1
            if np.isnan(sum_months.loc[id, 201701]):
                total_months += 1
                if np.isnan(sum_months.loc[id, 201612]):
                    total_months += 1
                    if np.isnan(sum_months.loc[id, 201611]):
                        total_months += 1
                        if np.isnan(sum_months.loc[id, 201610]):
                            total_months += 1
    no_songs.append(total_months)

sum_months['no_songs_cp6_test'] = no_songs

## CREATE clusters on median seconds monthly (need to fill NAs with 0s)
tmonths = pd.pivot_table(ul, values='logsecs', index=['new_id'], columns=['yearMonth'], aggfunc=np.median)
tmonths.fillna(0, inplace = True)

from sklearn.cluster import KMeans
tmonths_temp = tmonths.copy(deep=True)

kmeans = KMeans(n_clusters=9, random_state=0).fit(tmonths)
tmonths_temp['cluster9'] = kmeans.labels_
ul_cluster9 = tmonths_temp.reset_index()[['new_id','cluster9']]

##CREATE VARIABLE TOTAL ENTRIES
ul_entries = ul.groupby('new_id')['new_id'].count().reset_index(name='total_entries')

#CREATE FREQUENCY USING MEAN
ul_freq2 = ul.groupby('new_id')['time_lapsed'].mean().reset_index(name='freq_days_mean')

#CREATE UL TENURE
ul_max_date = ul.groupby('new_id')['date'].max().reset_index(name = 'max_date')
ul_min_date = ul.groupby('new_id')['date'].min().reset_index(name = 'min_date')

ul_tenure = ul_max_date.merge(ul_min_date, on='new_id', how='inner')
ul_tenure['ul_tenure'] = (ul_tenure['max_date'] - ul_tenure['min_date']).astype('timedelta64[D]')
ul_tenure.drop(['max_date','min_date'], axis = 1, inplace = True)

#MERGE TO TRAIN DATASET
f_ul2 = train.merge(sum_months.reset_index(), left_on='new_id', right_on='new_id', how='left', copy = False)[
    ['new_id','is_churn','no_songs_cp6_train','months_listening','listening_p6_train','listening_p12_train','logavg_secs_p12_train']]\
        .merge(ul_freq, left_on = 'new_id', right_on = 'new_id', how='left', copy=False)\
        .merge(ul_csecs, left_on = 'new_id', right_on = 'new_id', how='left', copy=False)\
        .merge(ul_entries, left_on = 'new_id', right_on = 'new_id', how='left', copy=False)\
        .merge(ul_cluster9, on="new_id", how='left')\
        .merge(ul_freq2, on="new_id", how='left')\
        .merge(ul_tenure, on='new_id', how='left')

#MERGE TO TEST DATASET
f_test = test.merge(sum_months.reset_index(), left_on='new_id', right_on='new_id', how='left', copy = False)[
    ['new_id','is_churn','no_songs_cp6_test','months_listening','listening_p6_test','listening_p12_test','logavg_secs_p12_test']]\
        .merge(ul_freq, left_on = 'new_id', right_on = 'new_id', how='left', copy=False)\
        .merge(ul_csecs, left_on = 'new_id', right_on = 'new_id', how='left', copy=False)\
        .merge(ul_entries, left_on = 'new_id', right_on = 'new_id', how='left', copy=False)\
        .merge(ul_cluster9, on="new_id", how='left')\
        .merge(ul_freq2, on="new_id", how='left')\
        .merge(ul_tenure, on='new_id', how='left')

f_ul2.columns = ['new_id', 'is_churn', 'no_songs_cp6', 'months_listening',
       'listening_p6', 'listening_p12', 'logavg_secs_p12',
       'freq_days', 'std_logsecs', 'total_entries', 'cluster9',
       'freq_days_mean', 'ul_tenure']

#np.sum(pd.isnull(f_ul2))

f_test.columns = ['new_id', 'is_churn', 'no_songs_cp6', 'months_listening',
       'listening_p6', 'listening_p12', 'logavg_secs_p12',
       'freq_days', 'std_logsecs', 'total_entries', 'cluster9',
       'freq_days_mean', 'ul_tenure']

#np.sum(pd.isnull(f_test))

f_ul2['no_songs_cp6'] = f_ul2['no_songs_cp6'].fillna(6)
f_ul2['months_listening'] = f_ul2['months_listening'].fillna('0-6')
f_ul2['listening_p6'] = f_ul2['listening_p6'].fillna(0)
f_ul2['listening_p12'] = f_ul2['listening_p12'].fillna(0)
f_ul2['total_entries'] = f_ul2['total_entries'].fillna(0)
f_ul2['ul_tenure'] = f_ul2['ul_tenure'].fillna(0)

f_ul2['logavg_secs_p12'] =f_ul2['logavg_secs_p12'].fillna(np.min(f_ul2['logavg_secs_p12']))

f_ul2['freq_days'] = 1/f_ul2['freq_days']
f_ul2['freq_days'] = f_ul2['freq_days'].fillna(0)

f_ul2['freq_days_mean'] = 1/f_ul2['freq_days_mean']
f_ul2['freq_days_mean'] = f_ul2['freq_days_mean'].fillna(0)

f_ul2['std_logsecs'] = f_ul2['std_logsecs'].fillna(0)

f_ul2['cluster9'] = f_ul2['cluster9'].fillna(np.max(f_ul2['cluster9'])+1)

#np.sum(pd.isnull(f_ul2))

f_test['no_songs_cp6'] = f_test['no_songs_cp6'].fillna(6)
f_test['months_listening'] = f_test['months_listening'].fillna('0-6')
f_test['listening_p6'] = f_test['listening_p6'].fillna(0)
f_test['listening_p12'] = f_test['listening_p12'].fillna(0)
f_test['total_entries'] = f_test['total_entries'].fillna(0)
f_test['ul_tenure'] = f_test['ul_tenure'].fillna(0)

f_test['logavg_secs_p12'] =f_test['logavg_secs_p12'].fillna(np.min(f_test['logavg_secs_p12']))

f_test['freq_days'] = 1/f_test['freq_days']
f_test['freq_days'] = f_test['freq_days'].fillna(0)

f_test['freq_days_mean'] = 1/f_test['freq_days_mean']
f_test['freq_days_mean'] = f_test['freq_days_mean'].fillna(0)

f_test['std_logsecs'] = f_test['std_logsecs'].fillna(0)

f_test['cluster9'] = f_test['cluster9'].fillna(np.max(f_test['cluster9'])+1)

#np.sum(pd.isnull(f_test))

#SAVE TO CSV
f_ul2.to_csv('../final_user_log_v2.csv', index = False)
f_test.to_csv('../final_user_log_test.csv', index = False)
