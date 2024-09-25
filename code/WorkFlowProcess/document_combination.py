import pandas as pd

#read each excel sheet into individuals data frames

df1 = pd.read_excel("Roomba Reviews.xlsx", sheet_name = 'iRobot Roomba 650')
df2 = pd.read_excel("Roomba Reviews.xlsx", sheet_name = 'iRobot Roomba 880')

#add columns to distinguish which record is from which group ##### DONT NEED THIS FOR PRACTICE DATA
df1['Group'] = 'Group 1'
df2['Group'] = 'Group 2'

#combined dataframes into one
df_total = pd.concat([df1,df2], ignore_index = True)

missing_values = df_total.isnull().sum()
print(missing_values)

#drop records with missing Title and Review
df_total.dropna(subset=['Title', 'Review'], how='any', inplace=True)



