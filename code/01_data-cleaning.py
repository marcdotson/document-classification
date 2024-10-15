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

