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
import pip
import contractions  # For expanding contractions (e.g., can't -> cannot)

from nltk.tokenize.toktok import ToktokTokenizer  # Toktok tokenizer for tokenization
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Spacy model download command (if not already installed)
import spacy  # For advanced NLP tasks
# Load Spacy language model
nlp = spacy.load('en_core_web_sm')

# Initialize Toktok tokenizer
tokenizer = ToktokTokenizer()

#_________________________________________________________________________

def clean_text_dataframe(df, drop_columns=None, text_columns=None):
    """
    ***** IF THERE ARE ANY OF THESE STEPS YOU DO NOT WANT TO PERFORM SIMPLY COMMENT THEM OUT*****

    Cleans a DataFrame by performing the following operations:
    1. Drops specified columns.
    2. Converts text in specified columns to lowercase and strips whitespace.
    3. Expands contractions.
    4. Removes punctuation from the text.
    5. Removes Stop Words from text
    6. Removes any numbers from text

    Parameters:
    df (pd.DataFrame): The input DataFrame to clean.
    drop_columns (list): List of columns to drop.
    text_columns (list): List of columns containing text to clean.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    #Drop specified columns
    if drop_columns:
        df_cleaned = df.drop(drop_columns, axis=1)
    else:
        df_cleaned = df.copy()

    #Convert text columns to lowercase and strip whitespace
    if text_columns:
        df_cleaned[text_columns] = df_cleaned[text_columns].apply(lambda x: x.str.lower().str.strip())

    #Expand contractions
        df_cleaned[text_columns] = df_cleaned[text_columns].apply(contractions.fix)

    #Remove punctuation
        df_cleaned[text_columns] = df_cleaned[text_columns].apply(lambda x: x.str.translate(str.maketrans('', '', string.punctuation)) if x.dtype == 'object' else x)
    
     #Remove digits of any length 
        df_cleaned[text_columns] = df_cleaned[text_columns].apply(
            lambda x: x.apply(lambda text: re.sub(r'\d+', '', text)) if x.dtype == 'object' else x
        )
    #Remove stop words from our text or most common words
    
    return df_cleaned


#________________________Example Usage__________________________

drop_columns = ['Col_1', 'Col_2']
df_combined  = "Insert your concatanated df here"
text_columns = ['Text_col_1', 'Text_col_2']

df_cleaned = clean_text_dataframe(df_combined, drop_columns=drop_columns, text_columns=text_columns)


'''
once text has been cleaned move onto basic text statistics to determine if you should be returning to 
clean text more or filter out more stop words
'''

