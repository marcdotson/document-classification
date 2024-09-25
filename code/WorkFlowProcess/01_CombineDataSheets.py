import pandas as pd

def load_and_combine_data(file_path, labels):
    """
    Loads data from all sheets in an Excel file, assigns group labels, and combines the data.

    Parameters:
    - file_path (str): Path to the Excel file.
    - labels (list of str): List of labels for each sheet, in the same order as the sheets.

    Returns:
    - combined_df (DataFrame): A DataFrame with combined data from all sheets and group labels.
    """
    # Load the Excel file with all sheet names ADJUST file type if needed
    data = pd.read_excel(file_path, sheet_name=None)

    # Ensure there are enough labels for all sheets
    if len(labels) != len(data):
        raise ValueError("Number of labels must match the number of sheets.")

    # Combine the sheets, assigning the corresponding label to each
    combined_df_list = []
    for sheet_name, label in zip(data.keys(), labels):
        df = data[sheet_name]
        df['group'] = label
        combined_df_list.append(df)

    # Concatenate all DataFrames into one
    combined_df = pd.concat(combined_df_list, ignore_index=True)
    
    return combined_df

# Example usage
file_path = r'insert_file_path_here.xslx'
labels = ['Group One', 'Group Two']  # Make sure the number of labels matches the number of sheets
combined_data = load_and_combine_data(file_path, labels)

# Display the combined DataFrame
combined_data.head()




