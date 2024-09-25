{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "def load_and_combine_data(file_path, labels):\n",
    "    \"\"\"\n",
    "    Loads data from all sheets in an Excel file, assigns group labels, and combines the data.\n",
    "\n",
    "    Parameters:\n",
    "    - file_path (str): Path to the Excel file.\n",
    "    - labels (list of str): List of labels for each sheet, in the same order as the sheets.\n",
    "\n",
    "    Returns:\n",
    "    - combined_df (DataFrame): A DataFrame with combined data from all sheets and group labels.\n",
    "    \"\"\"\n",
    "    # Load the Excel file with all sheet names ADJUST file type if needed\n",
    "    data = pd.read_excel(file_path, sheet_name=None)\n",
    "\n",
    "    # Ensure there are enough labels for all sheets\n",
    "    if len(labels) != len(data):\n",
    "        raise ValueError(\"Number of labels must match the number of sheets.\")\n",
    "\n",
    "    # Combine the sheets, assigning the corresponding label to each\n",
    "    combined_df_list = []\n",
    "    for sheet_name, label in zip(data.keys(), labels):\n",
    "        df = data[sheet_name]\n",
    "        df['group'] = label\n",
    "        combined_df_list.append(df)\n",
    "\n",
    "    # Concatenate all DataFrames into one\n",
    "    combined_df = pd.concat(combined_df_list, ignore_index=True)\n",
    "    \n",
    "    return combined_df\n",
    "\n",
    "# Example usage\n",
    "file_path = r'insert_file_path_here.xslx'\n",
    "labels = ['Group One', 'Group Two']  # Make sure the number of labels matches the number of sheets\n",
    "combined_data = load_and_combine_data(file_path, labels)\n",
    "\n",
    "# Display the combined DataFrame\n",
    "combined_data.head()\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
