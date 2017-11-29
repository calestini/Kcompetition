import numpy as np
import pandas as pd
import modeling as mdl
import seaborn as sns

new_id = pd.read_csv('../new_ids_v2.csv')

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, log_loss

test_merged = mdl.read_datasets(['final_txn_test_v2', 'final_user_log_test'])
test_merged = mdl.prep_variables(test_merged)

train = mdl.read_datasets()
train = mdl.prep_variables(train)

X_train, X_test, y_train, y_test = mdl.train_test(train, test_size=0, oversampling=0)

test_merged.drop('is_churn', axis=1, inplace=1)

# Random Forest Grid Search, max depth of 23 seems ideal
parameters = [{"max_depth": []}]

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

forest = RandomForestClassifier(n_estimators= 100, criterion = 'entropy', max_depth = 19)
forest.fit(X_train, y_train)
y_pred = forest.predict(test_merged)
y_prob = forest.predict_proba(test_merged)
y_prob_train = forest.predict_proba(X_train)

print('Random Forest Train \t{:.4f}' .format(log_loss(y_train, y_prob_train)))

test_merged['is_churn'] = y_prob[:,1] 
test_merged['new_id'] = test_merged.index
test_f = test_merged[['new_id','is_churn']].merge(new_id, on='new_id', how='inner')
test_f.drop('new_id', axis=1, inplace=True)
test_f.to_csv('../test_prediction.csv', index=False)


#XGBOOST
import xgboost as xgb

def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'log_loss', log_loss(labels, preds)

params = {
    'eta': 0.002, 
    'max_depth': 20,
    'min_child_weight': 5,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'silent': True
}

cols = [c for c in train.columns if c not in 'is_churn']

X_train, X_test, y_train, y_test = train_test_split(train[cols], train['is_churn'], test_size=0.3)
watchlist = [(xgb.DMatrix(X_train, y_train), 'train'), (xgb.DMatrix(X_test, y_test), 'valid')]
model = xgb.train(params, xgb.DMatrix(X_train, y_train), 1500,  watchlist, feval=xgb_score, maximize=False, verbose_eval=50, early_stopping_rounds=50) 

pred = model.predict(xgb.DMatrix(test_merged[cols]), ntree_limit=model.best_ntree_limit)

test['is_churn'] = pred.clip(0.0000001, 0.999999)
test[['msno','is_churn']].to_csv('submission3.csv.gz', index=False, compression='gzip')