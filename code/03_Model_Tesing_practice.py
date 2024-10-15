# Import necessary libraries
import nltk
import re
from nltk.corpus import brown
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import string  # For string operations
import numpy as np
import contractions  # For expanding contractions (e.g., can't -> cannot)
from sklearn.svm import SVC
import json

from nltk.tokenize.toktok import ToktokTokenizer  # Toktok tokenizer for tokenization
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Spacy model download command (if not already installed)
import spacy  # For advanced NLP tasks
# Load Spacy language model
nlp = spacy.load('en_core_web_sm')

# Initialize Toktok tokenizer
tokenizer = ToktokTokenizer()
lemmatizer = WordNetLemmatizer()  # Initialize the lemmatizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Download the nltk features
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

#----------------------------------------------------------------------------------------------------------------

#read in our data
test_data = pd.read_excel(r'C:\Users\ksbuf\OneDrive\Desktop\Invista PRoject\document-classification\data\test_data_roomba.xlsx')
training_data = pd.read_excel(r'C:\Users\ksbuf\OneDrive\Desktop\Invista PRoject\document-classification\data\yelp_review_subset.xlsx')

#Evaluate our class distribution prior to analyzing anything
class_counts = training_data['stars'].value_counts()
print(class_counts)

#drop null values and convert our training data from yelp to same format from roomba data (Five star, not five star)
training_data = training_data.dropna()
training_data['stars'] = training_data['stars'].replace({1 : 0, 2 :0, 3: 0, 4:0, 5: 1})


#Evaluate our class distribution prior to analyzing anything
class_counts = training_data['stars'].value_counts()
print(class_counts)

#----------------------------------------------------------------------------------------------------------------
# Establish our vecotrizer
vectorizer = TfidfVectorizer(min_df = 5 , max_df = .6, ngram_range=(1, 2))

# Split data into text and labels
texts = training_data['text'].tolist()
labels = training_data['stars'].tolist()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size = 0.2, random_state = 42)

#Split out our training data and see if the predictorvector shapes match correctly

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

num_features_train = X_train_vec.shape
num_features_test = X_test_vec.shape

print(f"Number of features in the training set: {num_features_train}")
print(f"Number of features in the test set: {num_features_test}")
print("THESE MUST HAVE THE EXACT SAME NUMBER OF PREDICTOR VAIRABLES TO BE COMPLETE TO MAKE PREDICTIONS")

#----------------------------------------------------------------------------------------------------------------

#create just some basic models to see how each performs prior to any tuning
models = {
    'Naive Bayes': MultinomialNB(),
    'Linear SVM': LinearSVC(),
    # 'Random Forests': RandomForestClassifier() #This takes about 5 minutes to run uncomment to run if you want
}

# Perform cross-validation for each model
for name, model in models.items():
    scores = cross_val_score(model, X_train_vec, y_train, cv = 5)
    print(f"{name} Cross-Validation Mean Accuracy: {scores.mean():.4f}")


#Hyper Parameter Tuning
param_grid_svc = {
    'C': [0.001, 0.01, 0.1, 1, 10],  # Expanded C values for better regularization exploration
    'loss': ['hinge', 'squared_hinge'],    # Loss functions
    'max_iter': [1000, 5000, 10000],     # Iteration limits
}

# Perform the Grid search for LinearSVC 
grid_search_svc = GridSearchCV(LinearSVC(random_state= 42), param_grid_svc, cv = 5, verbose =10)
grid_search_svc.fit(X_train_vec, y_train)

# After fitting, you would typically print the best parameters as follows:
print("Best parameters for LinearSVC:", grid_search_svc.best_params_)

#----------------------------------------------------------------------------------------------------------------

#test out our model on the test data and see how it performs

y_pred = grid_search_svc.predict(X_test_vec)

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

#print out the classification Report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#----------------------------------------------------------------------------------------------------------------

#read in the actual labeled section of the roomba dataset to compare how our model labels them with how they are really labeled

test_df_labeled = pd.read_excel(r'C:\Users\ksbuf\OneDrive\Desktop\Invista PRoject\document-classification\data\training_data_roomba.xlsx')
test_df_labeled = test_df_labeled.dropna()

# vectorize our unlabeled reviews
test_df_vec = vectorizer.transform(test_df_labeled['Review'].dropna())
num_features = test_df_vec.shape
#check to make sure that the features match the training features
num_features


#create our predictions and apply it to the test DF
predicted_labels = grid_search_svc.predict(test_df_vec)
for text, label in zip(test_df_labeled['Review'], predicted_labels):
    test_df_labeled['predicted_label'] = predicted_labels


#go through and see if the model misclassified any of the reviews compared with how it was originally labeled
mis_labeled  = []
for index, row in test_df_labeled.iterrows():
    if row['predicted_label'] != row['Received Five Stars']:
        mis_labeled.append(row)


#print out the amount that were mislabeled
print(f'Out of {len(test_df_labeled)} rows to be labeled, {len(mis_labeled)} were mislabeled. '
      f'Meaning that {(1 - (len(mis_labeled) / len(test_df_labeled))) * 100:.2f}% were correctly classified.')



#spot check the mislabeled items and see where it may be going wrong
for row in range (1,50):
  print(f"Review: {mis_labeled[row]['Review']}")
  print(f"Predicted label: {mis_labeled[row]['predicted_label']}")
  print(f"Actual label: {mis_labeled[row]['Received Five Stars']}")
  print("\n")


