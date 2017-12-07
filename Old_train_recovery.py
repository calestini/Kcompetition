# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 22:56:51 2017

@author: Cathy
"""


import numpy as np
import pandas as pd
import modeling as mdl
import seaborn as sns
import matplotlib.pyplot as plt
import data_manipulation_v2 as dm

new_id = pd.read_csv('../new_ids.csv')
new_id_v2 = pd.read_csv('../new_ids_v2.csv')

#Assign msnos back to train files
train = dm.train()
train_v2 = dm.train_v2()
test = dm.test_v2()

train = train.merge(new_id, on = 'new_id', how = 'inner')
train_v2 = train_v2.merge(new_id_v2, on = 'new_id', how = 'inner')
test = test.merge(new_id_v2, on = 'new_id', how = 'inner')

#Determine overlap in train file
train['in_v2'] = train['msno'].isin(train_v2['msno'])
train_v2['in_og'] = train_v2['msno'].isin(train['msno'])
train['in_test'] = train['msno'].isin(test['msno'])

print(train.in_v2.value_counts())
print(train_v2.in_og.value_counts())
print(train.in_test.value_counts())

train_feb_churners = train[train['in_v2']==False]



