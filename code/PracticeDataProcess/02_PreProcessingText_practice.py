#!/usr/bin/env python
# coding: utf-8

# In[45]:


'''
Importing all required libraries for text cleaning.
Includes libraries for text processing, web scraping, tokenization, and more.
'''

import re  # For regular expressions
import string  # For string operations
import nltk  # For natural language processing
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup  # For web scraping (if needed)
import contractions  # For expanding contractions (e.g., can't -> cannot)
import spacy  # For advanced NLP tasks
from nltk.tokenize.toktok import ToktokTokenizer  # Toktok tokenizer for tokenization
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
# Spacy model download command (if not already installed)
# !python -m spacy download en_core_web_sm

# Load Spacy language model
nlp = spacy.load('en_core_web_sm')

# Initialize Toktok tokenizer
tokenizer = ToktokTokenizer()

#read each excel sheet into individuals data frames

df1 = pd.read_excel(r"C:\Users\ksbuf\OneDrive\Desktop\Invista PRoject\document-classification\data\Roomba Reviews.xlsx", sheet_name = 'iRobot Roomba 650')
df2 = pd.read_excel(r"C:\Users\ksbuf\OneDrive\Desktop\Invista PRoject\document-classification\data\Roomba Reviews.xlsx", sheet_name = 'iRobot Roomba 880')

#combined dataframes into one
df_combined = pd.concat([df1,df2], ignore_index = True)


df_cleaned = df_combined.drop(['Date'], axis = 1)

df_cleaned['Product'] = df_cleaned['Product'].replace({'iRobot Roomba 650 for Pets': '650', 'iRobot Roomba 880 for Pets and Allergies': '880'})


#Look for any null values in our reviews and see if they can be filled in with context from the title 
for title in df_cleaned['Title'][df_cleaned['Review'].isna()]:
    print(title)


#it appears that this could be a review itself rather than a title, the title appears to be "Truly a wonderful thing." So we can make assumptions and fix this
df_cleaned[df_cleaned['Review'].isna()].head()

#split up the title and the review
df_cleaned.loc[240, 'Title'] = 'Truly a wonderful thing.'
df_cleaned.loc[240, 'Review'] = 'Reminded me of that old Peter, Paul & Mary song, Marvelous Toy." Truly a wonderful thing.'

#check to make sure the values are correct
df_cleaned.loc[240].head()
df_cleaned['Review'].isna().sum()

#change any column names 
df_cleaned['Received Five Stars'] = df_cleaned['Rating']
df_cleaned = df_cleaned.drop('Rating', axis = 1)
df_cleaned['Received Five Stars'] = df_cleaned['Received Five Stars'].replace({'Five Stars': 1, 'Not Five Stars': 0})


# Apply lower() and strip() to both 'Title' and 'Review' columns
df_cleaned[['Title', 'Review']] = df_cleaned[['Title', 'Review']].apply(lambda x: x.str.lower().str.strip())


#split up our contractions prior to us going through and removing the punctuation 
df_cleaned['Title'] = df_cleaned['Title'].fillna('').apply(contractions.fix)
df_cleaned['Review'] = df_cleaned['Review'].apply(contractions.fix)


#remove all of our punctionation
df_cleaned['Title'] = df_cleaned['Title'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if isinstance(x, str) else x)
df_cleaned['Review'] = df_cleaned['Review'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if isinstance(x, str) else x)

# Remove extra spaces from the 'Title' and 'Review' columns
df_cleaned[['Title', 'Review']] = df_cleaned[['Title', 'Review']].apply(lambda x: x.str.split().str.join(' '))

# remove any stop words from our text columns
# Tokenize the text in each row of the column, remove stopwords, and join the tokens back
stop_words = set(stopwords.words('english'))

# Apply the stopword removal to the 'Review' column without a function
df_cleaned['Review'] = df_cleaned['Review'].apply(
    lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))

#create a function to lemmatize the text if needed
def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    return lemmatized_text

#apply function
df_cleaned['Review'] = df_cleaned['Review'].apply(lemmatize_text)
df_cleaned['Title'] = df_cleaned['Title'].apply(lemmatize_text)


#check for top 20 most common words and see if we need to create a unique stop words list to drop these words
def word_frequency(text, N):
    tokens = word_tokenize(text)  # Tokenizing text into words
    frequency = Counter(tokens)  # Calculating the frequency of each word
    return frequency.most_common(N)  # Returning the top N most frequent words

text = ' '.join(df_cleaned['Review'].astype(str))
top_words = word_frequency(text, 20)

for word in top_words:
    print(word)


# create a list with the additional stop words that we should remove or would not be super useful

more_stop_words = [ 'roomba', 'get', 'go', 'thing', 'like']

df_cleaned['Review'] = df_cleaned['Review'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in more_stop_words]))
#check for top 20 most common words and see if we need to create a unique stop words list to drop these words
def word_frequency(text, N):
    tokens = word_tokenize(text)  # Tokenizing text into words
    frequency = Counter(tokens)  # Calculating the frequency of each word
    return frequency.most_common(N)  # Returning the top N most frequent words

text = ' '.join(df_cleaned['Review'].astype(str))
top_words = word_frequency(text, 20)

for word in top_words:
    print(word)

# create a combined text column if we would like to use it at a later time
df_cleaned['All text'] = df_cleaned['Title'] + ' ' + df_cleaned['Review']



#spit out our data that will be used to train the model and the data that we will then test the model on

# Creating DataFrame for test data where 'Received Five Stars' is NaN
df_test_data = df_cleaned[df_cleaned['Received Five Stars'].isna()]

# Creating DataFrame for training data where 'Received Five Stars' is not NaN
df_training_data = df_cleaned[df_cleaned['Received Five Stars'].notna()]

# Getting the shapes of both DataFrames
test_data_shape = df_test_data.shape
training_data_shape = df_training_data.shape

test_data_shape, training_data_shape

#SAVE EACH FILE TYPE

df_cleaned.to_excel(r"C:\Users\ksbuf\OneDrive\Desktop\Invista PRoject\document-classification\data\cleaned_data_roomba.xlsx", index=False)
df_test_data.to_excel(r"C:\Users\ksbuf\OneDrive\Desktop\Invista PRoject\document-classification\data\test_data_roomba.xlsx", index=False)
df_training_data.to_excel(r"C:\Users\ksbuf\OneDrive\Desktop\Invista PRoject\document-classification\data\training_data_roomba.xlsx", index=False)
