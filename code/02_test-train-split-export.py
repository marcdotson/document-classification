import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#_______________________________________________________________________________________________
#BELOW IS THE CODE TO SPLIT OUR CLEANED DATA INTO LABELED DATA, PERFORM TRAIN/TEST SPLIT AND 
# THEN STORE TRAIN/ TEST INDICIES THAT CAN BE CALLED DURING MODEL TESTING
#_______________________________________________________________________________________________

# Reading the csv file into a DataFrame
df_cleaned = pd.read_csv(r'data/READ_CLEAN_FILE_PATH.CSV')

# Filter labeled data for the specified values
df_labeled = df_cleaned[df_cleaned['Target_col'].isin(['Var1', 'Var2'])]

# Perform the train-test split
train_data, test_data = train_test_split(df_labeled, test_size=0.2, random_state=42)

# Save the indices to ensure consistency
train_indices = train_data.index
test_indices = test_data.index

# Export the indices as a CSV file for reuse
indices_df = pd.DataFrame({'train_indices': pd.Series(train_indices), 'test_indices': pd.Series(test_indices)})
indices_df.to_csv(r'data/train_test_indices.csv', index=False)


