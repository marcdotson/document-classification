























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
