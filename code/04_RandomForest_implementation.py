# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

#______________________________________________________________________________

#Load in the data
training_data = pd.read_excel('Insert file path here')
training_data = training_data.dropna()

#create the vector tranformation
vectorizer = TfidfVectorizer(min_df = 2 , max_df = .8, ngram_range=(1,2))

# Split data into text and labels
texts = training_data['TEXT COL'].tolist()
labels = training_data['Targer_Var'].tolist()

# Split data into training and testing sets, specify the test size you would like
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size = 0.2, random_state = 42)

#transform our X train/test split into vectors for analysis later
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#verify that the vector transformation was succesful
print(f'Number of features in training split: {X_train_vec.shape}')
print(f'Number of features in testing split: {X_test_vec.shape}')

#___________________________________________________________________________

# Define the parameter grid for RandomForestClassifier
param_grid_rf = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [None, 10, 20, 30],  # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples required at a leaf node

}

# Grid search for Random Forest
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state= 42, class_weight ='balanced'), param_grid_rf, cv = 5, verbose =10)
grid_search_rf.fit(X_train_vec, y_train)
# After fitting, you would typically print the best parameters as follows:
print("Best parameters for LinearSVC:", grid_search_rf.best_params_)

#________________________________________________________________________

# Testing the best estimator on the test set
best_model = grid_search_rf.best_estimator_
y_pred = best_model.predict(X_test_vec)

#see the results
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
