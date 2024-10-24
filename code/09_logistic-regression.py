import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np


#parameter grid for Logistic Regression
param_dist = {
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
    'C': np.logspace(-3, 2, 100),  # Regularization strength from 0.001 to 100
    'class_weight': ['balanced']  # Penalize based on class distribution
}

#Randomized search with cross-validation
random_search = RandomizedSearchCV(LogisticRegression(), param_distributions=param_dist, 
                                   n_iter=50, cv=5, scoring='f1', n_jobs=-1, verbose=1, random_state=42)

#Fit to the training data
random_search.fit(X_train_vec, y_train)

#Best parameters
print("Best parameters: ", random_search.best_params_)
