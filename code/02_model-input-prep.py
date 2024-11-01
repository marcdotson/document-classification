# Import necessary libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer


#_______________________________________________________________________________________________
#BELOW IS THE CODE TO SPLIT OUR DATA INTO LABELED AND UNLABELED
#_______________________________________________________________________________________________

# Reading the Excel file into a DataFrame
df_cleaned = pd.read_excel(r'READ_CLEAN_FILE_PATH.XLSX')

#Creating a dataframe that will store all of our unlabeled data
df_unlabeled = df_cleaned[df_cleaned['Independent_var_col'].isna()]

# Creating DataFrame for all of our labeled data to train the model
df_labeled = df_cleaned[df_cleaned['Independent_var_col'].notna()]

#Change our target class variables to 0 and 1 for model input
df_labeled['Targert Col'] = df_labeled['Targert Col'].repalce([{'Var1': 0, 'Var2': 1}])


#UNCOMMENT THE CODE BELOW IF YOU WANT TO SPILT IT INTO SEPERATE EXCEL FILES 
# df_unlabeled.to_excel(r"INSERT NEW FILE PATH", index=False)
# df_labeled.to_excel(r"INSERT NEW FILE PATH", index=False)


#_______________________________________________________________________________________________
#BELOW IS THE CODE TO USE A TFIDF METHOD TO BE USED AS MODEL INPUT
#_______________________________________________________________________________________________

#create the vector tranformation CHANGE THE PARAMETERS AS NEEDED TO CHANGE NUMBER OF FEATURES THE MODEL WILL TAKE AS INPUT
tfidf_vec = TfidfVectorizer(min_df = 2 , max_df = .8, ngram_range=(1,2))

# Split labeled data into text and labels
texts = df_labeled['TEXT COL'].tolist()
labels = df_labeled['Targer_Var'].tolist()

# Split data into training and testing sets, specify the test size you would like
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size = 0.2, random_state = 42)

#transform our X train/test split into vectors for analysis later
X_train_vec = tfidf_vec.fit_transform(X_train)
X_test_vec = tfidf_vec.transform(X_test)

#verify that the vector transformation was succesful
print(f'Number of features in training split: {X_train_vec.shape}')
print(f'Number of features in testing split: {X_test_vec.shape}')

# Show value counts of the target variable in training and testing splits
print("\nValue counts for y_train:")
print(pd.Series(y_train).value_counts())

print("\nValue counts for y_test:")
print(pd.Series(y_test).value_counts())

#_______________________________________________________________________________________________
#BELOW IS THE CODE TO USE A BASIC COUNT VECTORIZER METHOD TO BE USED AS MODEL INPUT
#_______________________________________________________________________________________________
 
#create the vector tranformation CHANGE THE PARAMETERS AS NEEDED TO CHANGE NUMBER OF FEATURES THE MODEL WILL TAKE AS INPUT
count_vec = CountVectorizer(min_df = 2 , max_df = .8, ngram_range=(1,2))

# Split labeled data into text and labels
texts = df_labeled['TEXT COL'].tolist()
labels = df_labeled['Targer_Var'].tolist()

# Split data into training and testing sets, specify the test size you would like
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size = 0.2, random_state = 42)

#transform our X train/test split into vectors for analysis later
X_train_vec = count_vec.fit_transform(X_train)
X_test_vec = count_vec.transform(X_test)

#verify that the vector transformation was succesful
print(f'Number of features in training split: {X_train_vec.shape}')
print(f'Number of features in testing split: {X_test_vec.shape}')

# Show value counts of the target variable in training and testing splits
print("\nValue counts for y_train:")
print(pd.Series(y_train).value_counts())

print("\nValue counts for y_test:")
print(pd.Series(y_test).value_counts())


#_______________________________________________________________________________________________
#BELOW IS THE CODE TO USE WORD EMBEDDINGS AS THE INPUT INTO THE MODEL
#_______________________________________________________________________________________________
 
#bring in model from HuggingFace
model = SentenceTransformer("ENTER THE HUGGING FACE EMBEDDINGS MODEL YOU NEED HERE")

#encode model
embeddings = model.encode(df_cleaned['text_column'])

#Add the embeddings to our cleaned dataframe as a new column
df_cleaned['Embeddings'] = list(embeddings)

#Export the dataframe with the embeddings so this does not have to be run everytime, Once data is exported comment this code out
df_cleaned.to_excel("INSERT NEW EMBEDDING FILE PATH HERE.XLSX", index = False)

#make a new copy of our df_cleaned in order to transfer over the new embeddings column
df_labeled = df_cleaned.copy()

# Split labeled data into embeddings and labels
embeddings = df_labeled['Embeddings'].tolist()  # Ensure this is a list/array
labels = df_labeled['Target_Var'].tolist()

# Split data into training and testing sets, specify the test size you would like
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Verify the shapes of the training and testing data to ensure it will input into the model
print(f'Number of training samples: {len(X_train)}')
print(f'Number of testing samples: {len(X_test)}')

# Show value counts of the target variable in training and testing splits
print("\nValue counts for y_train:")
print(pd.Series(y_train).value_counts())

print("\nValue counts for y_test:")
print(pd.Series(y_test).value_counts())

# If the embeddings are in a compatible format (like lists or numpy arrays), you can directly fit a model
# Convert to numpy arrays if necessary
X_train_array = np.array(X_train)
X_test_array = np.array(X_test)


