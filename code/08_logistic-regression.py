import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np


# Parameter grid for Logistic Regression
param_grid = {
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
    'C': np.logspace(-3, 2, 100),  # Regularization strength from 0.001 to 100
    'class_weight': ['balanced']  # Penalize based on class distribution
}

# Grid search with cross-validation
grid_search = GridSearchCV(LogisticRegression(class_weight='balanced'), param_grid=param_grid, 
                           cv=5, scoring='precision', n_jobs=-1, verbose=1)

#Best parameters
print("Best parameters for Linear Regression: ", grid_search.best_params_)
