# Importing all required libraries for text cleaning.
# Includes libraries for text processing, web scraping, tokenization, and more.

import re  # For regular expressions
import string  # For string operations
import nltk  # For natural language processing
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup  # For web scraping (if needed)
import contractions  # For expanding contractions (e.g., can't -> cannot)

from nltk.tokenize.toktok import ToktokTokenizer  # Toktok tokenizer for tokenization
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')  # For lemmatization

# Spacy model download command (if not already installed)
import spacy  # For advanced NLP tasks
# Load Spacy language model
nlp = spacy.load('en_core_web_sm')

# Initialize Toktok tokenizer
tokenizer = ToktokTokenizer()
lemmatizer = WordNetLemmatizer()  # Initialize the lemmatizer

#_________________________________________________________________________

def clean_text_dataframe(df, drop_columns=None, text_columns=None):
    """
    ***** IF THERE ARE ANY OF THESE STEPS YOU DO NOT WANT TO PERFORM SIMPLY COMMENT THEM OUT*****

    Cleans a DataFrame by performing the following operations:
    1. Drops specified columns.
    2. Converts text in specified columns to lowercase and strips whitespace.
    3. Expands contractions.
    4. Removes punctuation from the text.
    5. Removes stop words from text.
    6. Removes any numbers from text.
    7. Lemmatizes the words in the text.

    Parameters:
    df (pd.DataFrame): The input DataFrame to clean.
    drop_columns (list): List of columns to drop.
    text_columns (list): List of columns containing text to clean.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    # Drop specified columns
    if drop_columns:
        df_cleaned = df.drop(drop_columns, axis=1)
    else:
        df_cleaned = df.copy()

    # Convert text columns to lowercase and strip whitespace
    if text_columns:
        for col in text_columns:
            if col in df_cleaned.columns:
                # Apply cleaning steps only to string values
                df_cleaned[col] = df_cleaned[col].astype(str).str.lower().str.strip()
                
                # Expand contractions
                df_cleaned[col] = df_cleaned[col].apply(lambda x: contractions.fix(x) if isinstance(x, str) else x)

                # Remove punctuation
                df_cleaned[col] = df_cleaned[col].str.translate(str.maketrans('', '', string.punctuation))
                
                # Remove digits of any length
                df_cleaned[col] = df_cleaned[col].apply(lambda text: re.sub(r'\d+', '', text) if isinstance(text, str) else text)

                # Remove stop words
                stop_words = set(stopwords.words('english'))  # Set of English stop words
                df_cleaned[col] = df_cleaned[col].apply(
                    lambda text: ' '.join([word for word in text.split() if word not in stop_words]) if isinstance(text, str) else text
                )

                # Lemmatization
                df_cleaned[col] = df_cleaned[col].apply(
                    lambda text: ' '.join([lemmatizer.lemmatize(word) for word in text.split()]) if isinstance(text, str) else text
                )

    return df_cleaned


#_____________________________Example Usage_______________________________

#if more than one add extra columns seperated by a comma
drop_columns = ['Col_name']

#read in our dataframe that has been concat. to clean
df_combined = pd.read_excel(r'insert-file-path-here')

#=if more than one text column needs to be cleaned enter all seperated by commas
text_columns = ['text_col_name']

#create our clean df from the params above
df_cleaned = clean_text_dataframe(df_combined, drop_columns=drop_columns, text_columns=text_columns)

'''
Once text has been cleaned, move on to basic text statistics to determine if you should be returning to 
clean text more or filter out more stop words. One important thing to note is that 

'''

#Uncomment this code if you would like to export your data as a cleaned file

# df_cleaned.to_excel(r"new_file_path_here", index=False)