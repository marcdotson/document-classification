#Import our modules that are needed

import nltk
import re
from collections import Counter
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize

# Download NLTK resources
nltk.download('punkt')

#------------------------------------------------------------------------------------------------

# Reading the Excel file into a DataFrame
df_cleaned = pd.read_excel(r'C:\Users\ksbuf\OneDrive\Desktop\Invista PRoject\document-classification\data\cleaned_data_roomba.xlsx')

#------------------------------------------------------------------------------------------------

# Function to calculate word frequency
def word_frequency(text, N=20):
    tokens = word_tokenize(text.lower())  # Tokenizing and lowercasing text
    frequency = Counter(tokens)  # Calculating the frequency of each word
    return frequency.most_common(N)  # Returning the top N most frequent words

# Applying word frequency function to text columns
all_text = ' '.join(df_cleaned['insert_text_col'].astype(str))

#establish our top 20 words to see if we should create a new stop words list
top_words = word_frequency(all_text, N=20)

#print out our words and their counts
for word in top_words:
    print(word)


#--------------------------------------------------------------------------------------------

