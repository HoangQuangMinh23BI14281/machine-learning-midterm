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
    numerical_cols = [col for col in numerical_cols if col != 'Outcome'] # Exclude the 'Outcome' column from standardization
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

# Inverse standardization function to convert standardized values back to original values (no need to use for ann beacause it uses sigmoid function)
def inverse_standardize(df, stats, columns=None):

    df = df.copy()  
    if columns is None:  
        columns = df.columns.tolist()

    for col in columns:
        if col in stats:  
            mean = stats[col]['mean']
            std = stats[col]['std']
            if std != 0:  
                df[col] = df[col] * std + mean # Inverse standardization formula
            else:
                df[col] = mean  # Avoid division by zero

    return df

# Split the dataset into training and testing sets

def train_test_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    
    # Check if X is a DataFrame or a numpy array
    if isinstance(X, np.ndarray):
        indices = np.random.permutation(len(X))  # Generate random indices
    elif isinstance(X, pd.DataFrame):  # X is a pandas DataFrame
        indices = np.random.permutation(X.shape[0])  # Get random indices for DataFrame rows
    else:
        raise TypeError("X must be either a numpy array or a pandas DataFrame")

    split_idx = int(len(X) * (1 - test_size))  # Index for splitting
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    # Return train/test data based on indices
    if isinstance(X, np.ndarray):
        X_train, X_test = X[train_idx], X[test_idx]  # For numpy array
    elif isinstance(X, pd.DataFrame):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]  # For pandas DataFrame

    # For y (target variable), assuming y is a pandas Series or numpy array
    if isinstance(y, np.ndarray):
        y_train, y_test = y[train_idx], y[test_idx]
    elif isinstance(y, pd.Series):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test