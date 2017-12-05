import numpy as np
import pandas as pd
import modeling as mdl
import seaborn as sns
import matplotlib.pyplot as plt

new_id = pd.read_csv('../new_ids_v2.csv')

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, log_loss

train_mar = dm.train_v2()
train_feb = dm.train()

#What's overlapping?
train_feb_mar = train_feb.merge(train_mar, on = 'new_id', how = 'inner')
print(train_feb_mar['new_id'].nunique())

#What's in v2 that's not in og?
train_mar['in_feb'] = train_mar['new_id'].isin(train_feb['new_id'])

#What's in og but not v2?
train_feb['in_mar'] = train_feb['new_id']

#Churn in og and v2?

test_merged = mdl.read_datasets(['final_txn_test_v2', 'final_user_log_test'])
test_merged = mdl.prep_variables(test_merged)
test_merged['last_ar'] = test_merged['last_ar'].fillna(0)

train = mdl.read_datasets()
train = mdl.prep_variables(train)
train['last_ar'] = train['last_ar'].fillna(0)

#for c in train.columns:
 #   if train[c].isnull().sum()>0:
  #      print(train[c])

X_train, X_test, y_train, y_test = mdl.train_test(train, test_size=0, oversampling=0)

test_merged.drop('is_churn', axis=1, inplace=1)

#for c in test_merged.columns:
 #   if c not in train.columns:
  #      print(c)

# Random Forest Grid Search, max depth of 23 seems ideal
forest = RandomForestClassifier(n_estimators= 100, criterion = 'entropy', max_depth = 26)
forest.fit(X_train, y_train)
y_pred = forest.predict(test_merged)
y_prob = forest.predict_proba(test_merged)
y_prob_train = forest.predict_proba(X_train)

print('Random Forest Train \t{:.4f}' .format(log_loss(y_train, y_prob_train)))

test_merged['is_churn'] = y_prob[:,1] 
test_merged['new_id'] = test_merged.index
test_f = test_merged[['new_id','is_churn']].merge(new_id, on='new_id', how='inner')
test_f=test_f.groupby('msno')['is_churn'].mean().reset_index(name='is_churn')
test_f.to_csv('../test_prediction.csv', index=False)

test_f.shape

train.drop('is_churn', inplace = True, axis = 1)
features = pd.Series(data=forest.feature_importances_, index=train.columns)
features.sort_values(ascending=True, inplace=True)

plt.figure(num=None, figsize=(6, 30), dpi=80, facecolor='w', edgecolor='k')
plt.title('Feature Importances')
plt.barh(range(len(features)), features.values, color='b', align='center')
plt.yticks(range(len(features)), features.index) ## removed [indices]
plt.xlabel('Relative Importance')
plt.show()

sns.heatmap(train.corr())

########################################

# Random Forest Grid Search, max depth of 23 seems ideal
parameters = [{"max_depth": [20,21,22,23,24,25,26]}]

#Grid search, optimizing for log loss
grid_search = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 100, criterion = 'entropy'),
                           param_grid = parameters,
                           scoring = 'neg_log_loss',
                           cv = 5,
                           n_jobs = -1,
                           verbose = 1)
grid_search = grid_search.fit(X_train, y_train)

#Grid search results
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
grid_search.grid_scores_

###########################################


#XGBOOST
import xgboost as xgb

def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'log_loss', log_loss(labels, preds)

params = {
    'eta': 0.002, 
    'max_depth': 20,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'silent': True
}

cols = [c for c in train.columns if c not in 'is_churn']

X_train, X_test, y_train, y_test = train_test_split(train[cols], train['is_churn'], test_size=0.3)
watchlist = [(xgb.DMatrix(X_train, y_train), 'train'), (xgb.DMatrix(X_test, y_test), 'valid')]
model = xgb.train(params, xgb.DMatrix(X_train, y_train), 1500,  watchlist, feval=xgb_score, maximize=False, verbose_eval=50, early_stopping_rounds=50) 

pred = model.predict(xgb.DMatrix(test_merged[cols]), ntree_limit=model.best_ntree_limit)

test_merged['is_churn'] = pred.clip(0.0000001, 0.999999)

test_merged['new_id'] = test_merged.index
test_f = test_merged[['new_id','is_churn']].merge(new_id, on='new_id', how='inner')
test_f[['msno','is_churn']].to_csv('test_prediction.csv', index=False)