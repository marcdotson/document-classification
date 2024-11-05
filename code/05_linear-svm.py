# Import necessary libraries
import pandas as pd
from sklearn.svm import LinearSVC

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


#-----------------------------------------------------------------------------------------------
# Parameter tuning and model fitting
#-----------------------------------------------------------------------------------------------

# Define the parameter grid for Linear SVC Classifier we want to test these can be altered to best fit your needs

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],        # Expanded C values for better regularization exploration
    'loss': ['hinge', 'squared_hinge'],    # Loss functions
    'max_iter': [1000, 5000, 10000],       # Iteration limits
}

# Perform the Grid search for Linear SVM, store the best parameters
grid_search = GridSearchCV(LinearSVC(random_state= 42, class_weight ='balanced'), param_grid, cv = 5, verbose =10, scoring='precision')
grid_search.fit(X_train, y_train)
#store our best parameters
best_params = grid_search.best_params_

#print the best parameters as follows:
print("Best parameters for LinearSVC:", best_params)





