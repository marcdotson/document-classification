#import libraries
import re                                        #regular expressions
import string                                    #string operations
import nltk                                      #natural language processing
import pandas as pd                              #data frames
import numpy as np                               #arrays
import contractions                              #expanding contractions (e.g., can't -> cannot)
import spacy                                     #advanced NLP tasks
from nltk.tokenize.toktok import ToktokTokenizer #tokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter

#download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

#load stop words
stop_words = set(stopwords.words('english'))

#spacy model download command (if not already installed)
#!python -m spacy download en_core_web_sm

#load Spacy language model
nlp = spacy.load('en_core_web_sm')

#initialize Toktok tokenizer
tokenizer = ToktokTokenizer()

#read each excel sheet into individuals data frames
df1 = pd.read_excel('data/roomba-reviews.xlsx', sheet_name = 'iRobot Roomba 650')
df2 = pd.read_excel('data/roomba-reviews.xlsx', sheet_name = 'iRobot Roomba 880')

#add columns to distinguish which record is from which group ##### DONT NEED THIS FOR PRACTICE DATA
df1['Group'] = 'Group 1'
df2['Group'] = 'Group 2'

#combined dataframes into one
df_combined = pd.concat([df1,df2], ignore_index = True)

#change outcome column name and values to 1s and 0s
df_combined['Received Five Stars'] = df_combined['Rating']
df_combined = df_combined.drop('Rating', axis = 1)
df_combined['Received Five Stars'] = df_combined['Received Five Stars'].replace({'Five Stars': 1, 'Not Five Stars': 0})

#apply lower() and strip() to the 'Review' column
df_combined['Review'] = df_combined[['Review']].apply(lambda x: x.str.lower().str.strip())

#split up our contractions prior to us going through and removing the punctuation
df_combined['Review'] = df_combined['Review'].apply(contractions.fix)

#remove all of our punctionation
df_combined['Review'] = df_combined['Review'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if isinstance(x, str) else x)

#remove extra spaces from the 'Review' column
df_combined['Review'] = df_combined[['Review']].apply(lambda x: x.str.split().str.join(' '))

#aply the stop word removal to the 'Review' column without a function
df_combined['Review'] = df_combined['Review'].apply(
    lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))






#create a function to lemmatize the text if needed
def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    return lemmatized_text

#apply function
df_combined['Review'] = df_combined['Review'].apply(lemmatize_text)
df_combined['Title'] = df_combined['Title'].apply(lemmatize_text)


#check for top 20 most common words and see if we need to create a unique stop words list to drop these words
def word_frequency(text, N):
    tokens = word_tokenize(text)  # Tokenizing text into words
    frequency = Counter(tokens)  # Calculating the frequency of each word
    return frequency.most_common(N)  # Returning the top N most frequent words

text = ' '.join(df_combined['Review'].astype(str))
top_words = word_frequency(text, 20)

for word in top_words:
    print(word)


# create a list with the additional stop words that we should remove or would not be super useful

more_stop_words = [ 'roomba', 'get', 'go', 'thing', 'like']

df_combined['Review'] = df_combined['Review'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in more_stop_words]))
#check for top 20 most common words and see if we need to create a unique stop words list to drop these words
def word_frequency(text, N):
    tokens = word_tokenize(text)  # Tokenizing text into words
    frequency = Counter(tokens)  # Calculating the frequency of each word
    return frequency.most_common(N)  # Returning the top N most frequent words

text = ' '.join(df_combined['Review'].astype(str))
top_words = word_frequency(text, 20)

for word in top_words:
    print(word)

# create a combined text column if we would like to use it at a later time
df_combined['All text'] = df_combined['Title'] + ' ' + df_combined['Review']



#spit out our data that will be used to train the model and the data that we will then test the model on

# Creating DataFrame for test data where 'Received Five Stars' is NaN
df_test_data = df_combined[df_combined['Received Five Stars'].isna()]

# Creating DataFrame for training data where 'Received Five Stars' is not NaN
df_training_data = df_combined[df_combined['Received Five Stars'].notna()]

# Getting the shapes of both DataFrames
test_data_shape = df_test_data.shape
training_data_shape = df_training_data.shape

test_data_shape, training_data_shape

#SAVE EACH FILE TYPE

df_combined.to_excel(r"C:\Users\ksbuf\OneDrive\Desktop\Invista PRoject\document-classification\data\cleaned_data_roomba.xlsx", index=False)
df_test_data.to_excel(r"C:\Users\ksbuf\OneDrive\Desktop\Invista PRoject\document-classification\data\test_data_roomba.xlsx", index=False)
df_training_data.to_excel(r"C:\Users\ksbuf\OneDrive\Desktop\Invista PRoject\document-classification\data\training_data_roomba.xlsx", index=False)

