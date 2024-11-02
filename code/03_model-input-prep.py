# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

#_______________________________________________________________________________________________
#BELOW IS THE CODE TO LOAD OUR TRAIN/ TEST INDECIES AND SPECIFY OUR VARIABLES RUN THIS FIRST
#_______________________________________________________________________________________________

# Load train-test indices
indices_df = pd.read_csv('train_test_indices.csv')
train_indices = indices_df['train_indices'].dropna().astype(int).tolist()
test_indices = indices_df['test_indices'].dropna().astype(int).tolist()

# Load a cleaned/preprocessed dataset (e.g., with or without stop words)
df_cleaned = pd.read_csv('cleaned data version you need.csv')

# Filter labeled data for the specified values
df_labeled = df_cleaned[df_cleaned['Target_Col'].isin(['Var1', 'Var2'])]

# Change target class variables to 0 and 1 for model input
df_labeled['Target_Col'] = df_labeled['Target_Col'].replace({'Var1': 0, 'Var2': 1})

X = df_labeled['Text_Col']
y = df_labeled['Target_Col']

# Apply the saved indices to split the data
X_train, X_test = X.loc[train_indices], X.loc[test_indices]
y_train, y_test = y.loc[train_indices], y.loc[test_indices]

#_______________________________________________________________________________________________
#BELOW IS THE CODE TO USE A TFIDF METHOD TO BE USED AS MODEL INPUT, USE THIS CODE BLOCK TO IMPLEMENT
#_______________________________________________________________________________________________

#create the vector tranformation CHANGE THE PARAMETERS AS NEEDED TO CHANGE NUMBER OF FEATURES THE MODEL WILL TAKE AS INPUT
tfidf_vec = TfidfVectorizer(min_df = 2 , max_df = .8, ngram_range=(1,2))

#transform our X train/test split into vectors for analysis later
X_train = tfidf_vec.fit_transform(X_train)
X_test  = tfidf_vec.transform(X_test)

#verify that the vector transformation was succesful
print(f'Number of vector features in training split: {X_train.shape}')
print(f'Number of vector features in testing split: {X_test.shape}')


#_______________________________________________________________________________________________
#BELOW IS THE CODE TO USE A BASIC COUNT VECTORIZER METHOD TO BE USED AS MODEL INPUT, USE THIS CODE BLOCK TO IMPLEMENT
#_______________________________________________________________________________________________
 
#create the vector tranformation CHANGE THE PARAMETERS AS NEEDED TO CHANGE NUMBER OF FEATURES THE MODEL WILL TAKE AS INPUT
count_vec = CountVectorizer(min_df = 2 , max_df = .8, ngram_range=(1,2))

#transform our X train/test split into vectors for analysis later
X_train = count_vec.fit_transform(X_train)
X_test  = count_vec.transform(X_test)

#verify that the vector transformation was succesful
print(f'Number of vector features in training split: {X_train.shape}')
print(f'Number of vector features in testing split: {X_test.shape}')


#_______________________________________________________________________________________________
#BELOW IS THE CODE TO USE WORD EMBEDDINGS AS THE INPUT INTO THE MODEL, USE THIS CODE BLOCK TO IMPLEMENT
#_______________________________________________________________________________________________
 
 #-----------------------------------------------------------
#these first 5 steps are to get the embeddings and add them to our complete cleaned dataframe, once they are run comment them out to avoid running again

#Step 1: bring in model from HuggingFace
model = SentenceTransformer("Oillim/MiniLM-L6-v2")

#Step 2: encode model
embeddings = model.encode(df_cleaned['text_column'])

#Step 3: Store the embeddings in a dataframe
embeddings_df = pd.DataFrame(embeddings, columns = [f'embedding_{i}' for i in range(embeddings.shape[1])])

#Step 4: Add the embeddings to our cleaned dataframe as new columns
df_cleaned_embed = pd.concat([df_cleaned, embeddings_df], axis = 1 )

#Step 5: Export the dataframe with the embeddings 
df_cleaned_embed.to_csv("INSERT NEW EMBEDDING FILE PATH HERE.CSV", index = False)

#----------------------------------------------------------

#load in the embedings dataframe if the file exists already. Else run the steps above to create the new file. Comment this code out if not needed
df_cleaned_embed = pd.read_csv('INSERT EMBEDDINGS FILE PATH.CSV')

#create our labeled embeddings dataframe
df_labeled_embed = df_cleaned_embed[df_cleaned_embed['Target_Col'].isin(['Var1', 'Var2'])]

# Split labeled data into embeddings and labels
embeddings = df_labeled_embed[[col for col in df_labeled_embed.columns if col.startswith('embedding')]]
labels = df_labeled_embed['Target_col']

# Apply the saved indices to split the data for our embeddings dataframe
X_train, X_test = embeddings.loc[train_indices], embeddings.loc[test_indices]
y_train, y_test = labels.loc[train_indices], labels.loc[test_indices]

# If the embeddings are in a compatible format (like lists or numpy arrays), you can directly fit a model
# Convert to numpy arrays if necessary
X_train = np.array(X_train)
X_test = np.array(X_test)


