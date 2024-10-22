#imports needed in sagemaker
#!pip install sentence_transformers
#!pip install openpyxl

import pandas as pd
from sentence_transformers import SentenceTransformer

#read in cleaned dataframe, everything needs to be in one column
df = pd.read_excel('your file here')

#bring in model from HuggingFace
model = SentenceTransformer("Oillim/MiniLM-L6-v2")

#encode model
embeddings = model.encode(df)


#find similarities and print similarity matrix
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
print(similarities)



#EXAMPLE
# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("Oillim/MiniLM-L6-v2")

# sentences = ["That is a happy person",
#              "That is a happy dog",
#              "That is a very happy person",
#              "Today is a sunny day"]

# embeddings = model.encode(sentences)

# similarities = model.similarity(embeddings, embeddings)
# print(similarities)


#Sources:
#Huggingface Model: https://huggingface.co/Oillim/MiniLM-L6-v2?library=sentence-transformers
#Possible other models: https://huggingface.co/fibery/clustering-v2.0


