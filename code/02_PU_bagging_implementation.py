import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample

#----------------------------------------------------------------------------------

def pu_bagging(X, y, pos_label=1, n_estimators=10, base_estimator=None):
    
    # Create list to store classifiers
    classifiers = []
    
    # Loop through n_estimators
    for _ in range(n_estimators):
        # Sample from unlabeled data
        X_pos = X[y == pos_label]  # Positive Class
        X_unlabeled = X[y != pos_label]  # Unlabeled Class
        
        # Randomly sample unlabeled to create a pseudo-negative class
        X_unlabeled_sample, y_unlabeled_sample = resample(X_unlabeled, np.zeros(len(X_unlabeled)),
                                                          replace=True, n_samples=len(X_pos))

        # Combine positive and sampled unlabeled as a temporary dataset
        X_train = np.vstack([X_pos, X_unlabeled_sample])
        y_train = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_unlabeled_sample))])

        # Train classifier
        clf = base_estimator if base_estimator else DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        
        # Store classifier
        classifiers.append(clf)
    
    return classifiers

#----------------------------------------------------------------------------------------------

def pu_predict(X, classifiers):
    # Gather predictions from all classifiers
    predictions = np.zeros((len(X), len(classifiers)))
    
    for i, clf in enumerate(classifiers):
        predictions[:, i] = clf.predict(X)
    
    # Take the majority vote
    final_predictions = np.mean(predictions, axis=1)
    
    # Classify based on majority vote
    return (final_predictions >= 0.5).astype(int)


#----------------------------------Main Execution------------------------------------------------

df= "insert dataframe here"

documents = df['document'].values  # Extract the text data
labels = df['label'].values        # Extract the label column

#feature engineer X 
vectorizer = TfidfVectorizer("Change the parameters here if needed")
X = vectorizer.fit_transform(documents).toarray()

#change y to a numerical format if not already, it does not need to be transformed further
y = np.array([1 if label == 'CHANGE TO POSITIVE CLASS LABEL' else 0 for label in labels])


#Split the data, X is our features and Y is our labels ( split into 1 and 0 )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train PU Bagging model
classifiers = pu_bagging(X_train, y_train, pos_label=1, n_estimators=10)

# Predict on test data
y_pred = pu_predict(X_test, classifiers)

# Evaluate performance
print(classification_report(y_test, y_pred))