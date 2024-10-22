# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.svm import LinearSVC

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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

# Define the parameter grid for Linear SVC Classifier we want to test
param_grid_svc = {
    'C': [0.001, 0.01, 0.1, 1, 10],        # Expanded C values for better regularization exploration
    'loss': ['hinge', 'squared_hinge'],    # Loss functions
    'max_iter': [1000, 5000, 10000],       # Iteration limits
}

# Grid search for Linear SVM
grid_search_svc = GridSearchCV(LinearSVC(random_state= 42, class_weight ='balanced'), param_grid_svc, cv = 5, verbose =10)
grid_search_svc.fit(X_train_vec, y_train)

#print the best parameters as follows:
print("Best parameters for LinearSVC:", grid_search_svc.best_params_)


#-----------------------------------------------------------------------------------------------
# Model Evaluation
#-----------------------------------------------------------------------------------------------

# Testing the best estimator on the test set
best_model = grid_search_svc.best_estimator_
y_pred = best_model.predict(X_test_vec)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = sorted(list(set(y_test)))

# Plot the heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

#see the results
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


#-----------------------------------------------------------------------------------------------
# Predicting onto our test data and spot checking
#-----------------------------------------------------------------------------------------------

#create your test data here, BE SURE TO PERFORM THE SAME CLEANING STEPS TO THIS DATA AS THE TRAINING DATA
test_data = pd.read_excel(r'insert your test data here to test the model on')

#Vectorize your test data and print the shape to ensure that feature size is the same
test_df_vec = vectorizer.transform(test_data['Enter text column here'])
num_features = test_df_vec.shape
num_features


#create our predicted outcomes and add them onto our test data
predicted_labels = grid_search_svc.predict(test_df_vec)
for text, label in zip(test_data['REnter text column here'], predicted_labels):
    test_data['predicted_label'] = predicted_labels


#spot check some of the rows to see what we are dealing with
for row in range (1,50):
  print(f"Text: {test_data[row]['Text col here']}")
  print(f"Predicted label: {test_data[row]['predicted_label']}")
  print("\n")
