import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Take data from .csv file
data_frame = pd.read_csv('heart_attack_dataset.csv')
#print(data_frame)

# Check data types of each colums
print(data_frame.dtypes)

# Encode columns with "object" data type into numbers

categorical_cols = []

for col in data_frame.columns:
    unique_values = data_frame[col].nunique() # Check all the unique data in each column of the data frame (.nunique return unique values in column)
    if col != 'Outcome' and unique_values <=10: # Don't use outcome column and limit distinct value to 10 to limit columns with too many values ​​like age,...
      categorical_cols.append(col)

for col in categorical_cols:
    data_frame[col] = data_frame[col].astype('category') # Turn the data_type to category because only category type has .cat attribute and With large data, storing strings (objects) is very RAM consuming. Moreover,category only stores labels once and uses numeric codes to represent them in memory → significant savings.
    data_frame[col] = data_frame[col].cat.codes # Encode the string data into numbers

print(data_frame)

