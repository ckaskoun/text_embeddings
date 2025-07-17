"""
A simple file for printing the indices of different categories for human-coded
files. Categories must be manually named for filtering. Returns a list of
indices for each category.
"""

# Imports
import pandas as pd
import format_data as dt

# Create a dataframe of the coded scores
dict_codes = dt.format_coded("C:/Users/carlo/OneDrive/Documents/coded_memos") # Path to coded files goes here
df_codes = dt.combine_coded(dict_codes)
df_codes_weighted = dt.weight_coded(df_codes, 'category') # 'category' for highest level of coding

# Filter based on each category
filtered_df1 = df_codes_weighted[df_codes_weighted['Codes'].apply(lambda lst: 'No Code' in lst)]
filtered_df2 = df_codes_weighted[df_codes_weighted['Codes'].apply(lambda lst: 'Research Components' in lst)]
filtered_df3 = df_codes_weighted[df_codes_weighted['Codes'].apply(lambda lst: 'Direct Authenticity' in lst)]

# Print each row of the filtered DataFrame with its index
no_code_lst = []
rc_lst = []
da_lst = []

print('Direct Authenticity')
for index, row in filtered_df3.iterrows():
    da_lst.append(index)
    print(f"Index: {index}, Text: {df_codes_weighted['Sentences'][index]}")
print('\n')
print(da_lst)
print('\n')

print('Research Components')
for index, row in filtered_df2.iterrows():
    rc_lst.append(index)
    print(f"Index: {index}, Text: {df_codes_weighted['Sentences'][index]}")
print('\n')
print(rc_lst)
print('\n')

print('No Code')
for index, row in filtered_df1.iterrows():
    no_code_lst.append(index)
    print(f"Index: {index}, Text: {df_codes_weighted['Sentences'][index]}")
print('\n')
print(no_code_lst)
