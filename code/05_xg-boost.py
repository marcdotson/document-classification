import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd


# Create a DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

#model
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', class_weight ='balanced')

#hyperparameter grid
param_grid = {
    'max_depth': [3, 4, 5, 6, 7],
    'eta': [0.1, 0.3, 0.5],
    'n_estimators': [50, 100, 200]
}

#Create GridSearchCV object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='precision',  cv=5, verbose=1)
grid_search.fit(X_train, y_train)
#Best parameters and best score
best_params = grid_search.best_params_

print(f'Best Parameters for XGboost: {best_params}')


