# begin by importing the neccessary libraries that we will need

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

#--------------------------------------------------------------------------------------------

# Load in your data that you would like to train the model with and the data that you will be ulitmately testing the model on
test_data = pd.read_excel(r'file_path')
training_data = pd.read_excel(r'file_path')


# View your class distribution of the training data, this will help determine what our metric we should focus on should be
class_counts = training_data['insert y variable'].value_counts()
print(class_counts)


'''
- Here we will ensure that our data has already been preprocesed as we would like it to be and do any further cleaning that may have not been done prior.
- We can also at this point go through and change our outcome variable names to align with what we would like them to be throughout the analysis

-ex:

training_data = training_data.dropna()
training_data['stars'] = training_data['stars'].replace({1 : 0, 2 :0, 3: 0, 4:0, 5: 1})

'''


#-----------------------------------------------------------------------------------------------
# Feature Engineering/ vecorization
#-----------------------------------------------------------------------------------------------



''' Create a count vectorizer THIS IS WHERE YOU CAN DECIDE HOW YOU WANT TO FEATURE ENGINEER
        - NGRAM can be changed to include only one word combinations (1,1) or include one and two word combinations (1,2)
        - min_df refers to the minimum data freqeuncy for example 5 means it must appear in at least 5 documents,  .01 would refer to it must apear in 1% of documents to be included
        - Max_df refers to the maximum data frequency that must be met for it to be excluded. For example .9 = if the term occurs in 90% of the documents then discard it

'''

#create the vector tranformation
vectorizer = TfidfVectorizer(min_df = 'Change this to the param you want' , max_df = 'Change this to the param you want', ngram_range=('Change this to the param you want'))


# Split data into text and labels
texts = training_data['text'].tolist()
labels = training_data['stars'].tolist()


# Split data into training and testing sets, specify the test size you would like
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size = 0.2, random_state = 42)


#Split out our training data and see if the predictorvector shapes match correctly
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

num_features_train = X_train_vec.shape
num_features_test = X_test_vec.shape

#the shape of both of our sets should have the exact same number of features example output is shown below
'''
Number of features in the training set: (111999, 183390)
Number of features in the test set: (28000, 183390)
THESE MUST HAVE THE EXACT SAME NUMBER OF PREDICTOR VAIRABLES TO BE COMPLETE TO MAKE PREDICTIONS
'''

print(f"Number of features in the training set: {num_features_train}")
print(f"Number of features in the test set: {num_features_test}")
print("THESE MUST HAVE THE EXACT SAME NUMBER OF PREDICTOR VAIRABLES TO BE COMPLETE TO MAKE PREDICTIONS")


#-----------------------------------------------------------------------------------------------
# Model Selection
#-----------------------------------------------------------------------------------------------


#Create a loop that will run through and test your models and give a basic output of mean accuracy here is where you will specify the model you would like to use
models = {
    'Naive Bayes': MultinomialNB(),
    'Linear SVM': LinearSVC(),
    # 'Random Forests': RandomForestClassifier() #This takes about 5 minutes to run uncomment to run if you want
}


# Perform cross-validation for each model IF YOU NEED TO EVALUATE F1 SCORE YOU WILL NEED TO CHANGE THIS METRIC
for name, model in models.items():
    scores = cross_val_score(model, X_train_vec, y_train, cv = 5)
    print(f"{name} Cross-Validation Mean Accuracy: {scores.mean():.4f}")


'''
EXAMPLE USAGE OF F1 SCORE:

# Ensure your labels are encoded properly
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Define your models
models = {
    'Naive Bayes': MultinomialNB(),
    'Linear SVM': LinearSVC(),
    'Random Forest': RandomForestClassifier()
}

# Perform cross-validation and calculate F1 scores
for name, model in models.items():
    scores = cross_val_score(
        model, X_train_vec, y_train_encoded, cv=5,
        scoring=make_scorer(f1_score, average='weighted')  # Use weighted F1 score
    )
    print(f"{name} Cross-Validation Weighted F1 Score: {scores.mean():.4f}")

'''


# Once you decide which model is performing the best we can iterate through the possible parameters of that model and select the best fit, below is for linera SVC
param_grid_svc = {
    'C': [0.001, 0.01, 0.1, 1, 10],  # Expanded C values for better regularization exploration
    'loss': ['hinge', 'squared_hinge'],    # Loss functions
    'max_iter': [1000, 5000, 10000],     # Iteration limits
}
# Grid search for LinearSVC
grid_search_svc = GridSearchCV(LinearSVC(random_state= 42), param_grid_svc, cv = 5, verbose =10)
grid_search_svc.fit(X_train_vec, y_train)
# After fitting, you would typically print the best parameters as follows:
print("Best parameters for LinearSVC:", grid_search_svc.best_params_)



#-----------------------------------------------------------------------------------------------
# Model Evaluation
#-----------------------------------------------------------------------------------------------

#create our predictions 
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


#print a classification report and look at the metrics that will be the most beneficial
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
