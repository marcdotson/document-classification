import pandas as pd

#read each excel sheet into individuals data frames

df1 = pd.read_excel("insert_file_path.xslx", sheet_name = 'Group 1')
df2 = pd.read_excel("insert_file_path.xlsx", sheet_name = 'Group 2')

#add columns to distinguish which record is from which group
df1['Group'] = 'Group 1'
df2['Group'] = 'Group 2'

#combined dataframes into one
df_combined = pd.concat([df1,df2], ignore_index = True)