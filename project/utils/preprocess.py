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

def standardize(data_frame):
    df = data_frame.copy()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    stats = {}

    for col in numerical_cols:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std
        stats[col] = {'mean': mean, 'std': std}

    return df, stats

# Reuse mean and std from training set to standardize test set
def standardize_with_stats(data_frame, stats):
    df = data_frame.copy()

    for col in stats:
        mean = stats[col]['mean']
        std = stats[col]['std']
        if std != 0:
            df[col] = (df[col] - mean) / std
        else:
            df[col] = 0  # Avoid division by zero

    return df

# Preprocess the data frame by encoding and standardizing
def full_preprocess(data_frame):
    df_encoded = encode(data_frame)
    df_standardized, stats = standardize(df_encoded)
    return df_standardized, stats



# Split the dataset into training and testing sets
def train_test_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    split_idx = int(X.shape[0] * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
