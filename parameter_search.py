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

str_col = ['per_free_trial', 'txn_cnt'
           , 'per_lp_high', 'prev_churn_per', 'txn_median_gap', 'pmt_change_cnt'
           , 'lst_memb_expire_days', 'memb_tenure_days' , 'avg_daily_paid', 'last_ar'
           , 'list_actual_diff', 'payment_plan_days', 'end_lst_txn_days', 'last_cancel'
           , 'lst_pmt_plan_days', 'stopped_ar', 'lst_free_trial', 'not_equal', 'lst_memb_expire_post'
           , 'mean_ar', 'mean_cancel', 'missing_txns']

test = mdl.read_datasets(['final_txn_test_v2', 'final_user_log_test'])
test = mdl.prep_variables(test)

train = mdl.read_datasets()
train = mdl.prep_variables(train)
train_txn = train[train['missing_txns'] == False]
train_rest = train[train['missing_txns'] == True]
train_rest.drop(str_col, axis = 1, inplace = True)

X_train, X_test, y_train, y_test = mdl.train_test(train_txn, test_size=0, oversampling=0)

test.drop('is_churn', axis=1, inplace=1)
#for c in test_merged.columns:
 #   if c not in train.columns:
  #      print(c)

#Random forest model for users with transaction data
forest = RandomForestClassifier(n_estimators= 100, criterion = 'entropy', max_depth = 23)
forest.fit(X_train, y_train)
y_pred = forest.predict(test)
y_prob = forest.predict_proba(test)
y_prob_train = forest.predict_proba(X_train)

print('Random Forest Train \t{:.4f}' .format(log_loss(y_train, y_prob_train)))

#Assign predictions to test dataset
test['is_churn'] = y_prob[:,1] 
test['new_id'] = test.index
test_f = test[['new_id','is_churn']].merge(new_id, on='new_id', how='inner')
test_f.drop('new_id', axis = 1, inplace = True)
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