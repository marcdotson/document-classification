import pandas as pd
from openpyxl import load_workbook


#-----------------------------------------------------------------------------------------------
#Load and Filter Original Data
#-----------------------------------------------------------------------------------------------

# Load in the original data before cleaning
original_df = pd.read_csv('INSERT ORIGINAL DATA PATH HERE PRIOR TO CLEANING')

# Filter rows where our DataFrame includes only specified values in the target column
unlabeled_original_df = original_df[
    original_df['Targert_col'].isin(["Var1", "Var2"])
]

# Drop columns that are not needed for model performance tracking if needed

drop_columns = ['COLUMN_NAME_1', 'COLUMN_NAME_2']  # Replace with actual column names to drop
unlabeled_original_df.drop(columns=drop_columns, inplace=True)

model_eval_path = 'Insert the Existing file path here for our model evals.xlsx'

#-----------------------------------------------------------------------------------------------
#Load Existing or Initialize Model Performance Data
#-----------------------------------------------------------------------------------------------

# Try to load the existing Excel file, or create a new DataFrame if it doesn't exist
try:
    with pd.ExcelFile(model_eval_path) as reader:
        performance_df = pd.read_excel(reader, sheet_name='SPECIFY THE EXCEL SHEET NAME HERE')
except FileNotFoundError:
    # If the file doesn't exist, start with the filtered unlabeled original DataFrame
    performance_df = unlabeled_original_df.copy()

#-----------------------------------------------------------------------------------------------
# Add New Model Predictions and Probabilities
#-----------------------------------------------------------------------------------------------

# Define the model name, predictions, and probabilities for the new model
model_name = "Model 1 or whatever corresponds to model performance sheet"
predicted_labels = predicted_labels
prediction_probabilities = y_unlabled_prob

# Add columns for the model's predicted labels and probabilities
performance_df[f"{model_name}_Label"] = predicted_labels
performance_df[f"{model_name}_Prob"] = prediction_probabilities

#-----------------------------------------------------------------------------------------------
#Save Updated Data to Excel
#-----------------------------------------------------------------------------------------------

# Save the updated DataFrame to the Excel file, appending to existing data if needed
with pd.ExcelWriter(model_eval_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
    performance_df.to_excel(writer, index=False, sheet_name='SPECIFY THE EXCEL SHEET NAME HERE')

print(f"Added predicted labels and probabilities for {model_name} to {model_eval_path}")
