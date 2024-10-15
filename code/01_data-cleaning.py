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

#read each excel sheet into individuals data frames
df1 = pd.read_excel('data/Roomba Reviews.xlsx', sheet_name = 'iRobot Roomba 650')
df2 = pd.read_excel('data/Roomba Reviews.xlsx', sheet_name = 'iRobot Roomba 880')

#add columns to distinguish which record is from which group ##### DONT NEED THIS FOR PRACTICE DATA
df1['Group'] = 'Group 1'
df2['Group'] = 'Group 2'

#combined dataframes into one
df_combined = pd.concat([df1,df2], ignore_index = True)

#export combined dataset
# df_total.to_excel(r"C:\Users\ksbuf\OneDrive\Desktop\Invista PRoject\document-classification\data\combined_data_roomba.xlsx", index=False)
