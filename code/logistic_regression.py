import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

#Load your dataset from an Excel file
df = pd.read_excel('your_dataset.xlsx')

#Define features and target variable
X = df['Text_columns']  # Features: all columns except the target
y = df['target']  # Target: the column you're trying to predict

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#create the vector tranformation
vectorizer = TfidfVectorizer(min_df = 'Change this to the param you want' , max_df = 'Change this to the param you want', ngram_range=('Change this to the param you want'))

#transform our X train test split into vectors for analysis later
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#verify that the vector transformation was succesful THE NUMBER OF FEATURES IN EACH MUST MATCH 
print(f'Number of features in training split: {X_train_vec.shape}')
print(f'Number of features in testing split: {X_test_vec.shape}')

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
