# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Define the parameter grid for RandomForestClassifier change this as you need or want

param_grid_rf = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [None, 10, 20, 30],  # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples required at a leaf node

}
# Grid search for Random Forest
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state= 42, class_weight ='balanced'), param_grid_rf, cv = 5, verbose =10)
grid_search_rf.fit(X_train_vec, y_train)

#store our best parameters
best_params_rf = grid_search_rf.best_params_

#print the best parameters as follows:
print("Best parameters for LinearSVC:", best_params_rf)


