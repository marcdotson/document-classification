# Import necessary libraries

# from sklearn.feature_extraction.text import TfidfVectorizer #UNCOMMENT IF NEEDED 
# from sklearn.feature_extraction.text import CountVectorizer #UNCOMMENT IF NEEDED
import pandas as pd
import numpy as np

#-----------------------------------------------------------------------------------------------
# Predicting onto our unlabeled data and spot checking 
#-----------------------------------------------------------------------------------------------

#Import whatever cleaned dataset you have (embeddings or no embeddings) if needed, else comment this code out
df_cleaned = pd.read_csv(r'insert the cleaned file here.csv')

#specify the labels that are considered the "Unlabeled data" to filter the data out
df_unlabeled = df_cleaned[df_cleaned['Target_Col'].isin(['Var1', 'Var2'])]

#Comment this code out if you are not using a word vectorizer
unlabeld_text_vec = vectorizer.transform(df_unlabeled['Enter text column here']) #here you would use the same vectorizer you used to create model input, specify which you want to use

#Comment this code out if not using embeddings
unlabeld_text_emb = df_unlabeled[[col for col in df_unlabeled.columns if col.startswith('embedding')]]

#create our predicted outcomes and add them onto our test data be sure to specify if it is vectors or embeddings
predicted_labels = grid_search.predict('text_vec OR text_emb') #this can be our model from the grid search or best performing model, specify which you want to use

#convert our predicted labels by to their original values rather than 0 and 1
predicted_labels = np.where(predicted_labels == 0, "Var1", "Var2")

#Get our all of our predictions probabilties to store to be used in spot checking performance later
y_unlabeled_prob = grid_search.predict_proba('unlabeld_text_vec OR unlabeld_text_emb')[:, 1]

for text, label in zip(df_unlabeled['REnter text column here'], predicted_labels):
    df_unlabeled['predicted_label'] = predicted_labels


#spot check some of the rows to see what we are dealing with
for row in range (1,20):
  print(f"Text: {df_unlabeled[row]['Text col here']}")
  print(f"Predicted label: {df_unlabeled[row]['predicted_label']}")
  print("\n")

