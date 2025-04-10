import pandas as pd
import numpy as np

# Encode columns with "object" data type into numbers

def encode(data_frame):
    df = data_frame.copy() #avoid changing the original dataset
    categorical_cols = []

    for col in df.columns:
        unique_values = df[col].nunique() # Check all the unique data in each column of the data frame (.nunique return unique values in column)
        if unique_values <=10: # Don't use outcome column and limit distinct value to 10 to limit columns with too many values ​​like age,...
            categorical_cols.append(col)

    for col in categorical_cols:
        df[col] = df[col].astype('category') # Turn the data_type to category because only category type has .cat attribute and With large data, storing strings (objects) is very RAM consuming. Moreover,category only stores labels once and uses numeric codes to represent them in memory → significant savings.
        df[col] = df[col].cat.codes # Encode the string data into numbers

    return df




