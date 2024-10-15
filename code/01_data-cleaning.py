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

#change any column names 
df_combined['Received Five Stars'] = df_combined['Rating']
df_combined = df_combined.drop('Rating', axis = 1)
df_combined['Received Five Stars'] = df_combined['Received Five Stars'].replace({'Five Stars': 1, 'Not Five Stars': 0})

