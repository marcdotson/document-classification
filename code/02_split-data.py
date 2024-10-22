import pandas as pd


# Reading the Excel file into a DataFrame
df_cleaned = pd.read_excel(r'READ_CLEAN_FILE_PATH.XLSX')

#Creating a dataframe that will be used to test our model, IV's are null
df_test_data = df_cleaned[df_cleaned['Independent_var_col'].isna()]

# Creating DataFrame for training data where IV's is not NaN
df_training_data = df_cleaned[df_cleaned['Independent_var_col'].notna()]

# Getting the shapes of both DataFrames
test_data_shape = df_test_data.shape
training_data_shape = df_training_data.shape

print(f'Data with labeled IV to train model shape: {training_data_shape}')
print(f'Data with null IV values to test model on shape{test_data_shape}')

#------------------------------------------------------------------------------------------------
#UNCOMMENT THE CODE BELOW IF YOU WANT TO SPILT IT INTO SEPERATE EXCEL FILES 

# df_test_data.to_excel(r"New_test_data_path", index=False)
# df_training_data.to_excel(r"New_training_data_path", index=False)


