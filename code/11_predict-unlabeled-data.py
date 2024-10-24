# Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

#-----------------------------------------------------------------------------------------------
# Predicting onto our unlabeled data and spot checking 
#-----------------------------------------------------------------------------------------------

#BE SURE TO PERFORM THE SAME CLEANING STEPS TO THIS DATA AS THE LABELED DATA
unlabeled_data = pd.read_excel(r'insert your unlabeled data here to test the model on')

#BELOW WILL BE VARIOUS WAYS DEPENDING ON YOUR MODEL INPUT, WORD VECOTORS OR WORD EMBEDDINGS
unlabeld_text_vec = vectorizer.transform(unlabeled_data['Enter text column here']) #here you would use the same vectorizer you used to create model input, specify which you want to use
unlabeld_text_emb = unlabeled_data['Embeddings']



#create our predicted outcomes and add them onto our test data be sure to specify if it is vectors or embeddings
predicted_labels = grid_search.predict('text_vec OR text_emb') #this can be our model from the grid search or best performing model, specify which you want to use
for text, label in zip(unlabeled_data['REnter text column here'], predicted_labels):
    unlabeled_data['predicted_label'] = predicted_labels


#spot check some of the rows to see what we are dealing with
for row in range (1,50):
  print(f"Text: {unlabeled_data[row]['Text col here']}")
  print(f"Predicted label: {unlabeled_data[row]['predicted_label']}")
  print("\n")