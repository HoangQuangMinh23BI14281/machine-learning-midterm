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

def train_test_split(X, y, test_size=0.2, seed=50):
    np.random.seed(seed) # Set random seed for reproducibility
    
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

def load_data(data_path):
    data = pd.read_csv(data_path)
    data, stats = full_preprocess(data)
    features = [
        'Age', 'Gender', 'Cholesterol', 'BloodPressure', 'HeartRate', 'BMI', 'Smoker',
        'Diabetes', 'Hypertension', 'FamilyHistory', 'PhysicalActivity', 'AlcoholConsumption',
        'Diet', 'StressLevel', 'Ethnicity', 'Income', 'EducationLevel', 'Medication',
        'ChestPainType', 'ECGResults', 'MaxHeartRate', 'ST_Depression', 'ExerciseInducedAngina',
        'Slope', 'NumberOfMajorVessels', 'Thalassemia', 'PreviousHeartAttack', 'StrokeHistory',
        'Residence', 'EmploymentStatus', 'MaritalStatus'
    ]
    targets = ['Cholesterol', 'BloodPressure', 'HeartRate', 'BMI', 'Outcome']
    X = data[features]
    y = {}
    for target in targets:
        y[target] = data[target]
    return X, y, features, stats




# Explaining the code:
# 1. **encode**: This function encodes categorical columns into numerical values. It identifies columns with a limited number of unique values and converts them to category type, then encodes them as numbers.
# 2. **standardize**: This function standardizes numerical columns by subtracting the mean and dividing by the standard deviation. It also stores the mean and standard deviation for each column in a dictionary called `stats`.
# 3. **standardize_with_stats**: This function standardizes a DataFrame using pre-computed statistics (mean and standard deviation) from the training set. It avoids division by zero by checking if the standard deviation is not zero.
# 4. **full_preprocess**: This function combines the encoding and standardization steps. It first encodes the DataFrame and then standardizes it, returning both the standardized DataFrame and the statistics.
# 5. **inverse_standardize**: This function reverses the standardization process, converting standardized values back to their original values using the stored statistics. It allows for selective inverse standardization of specified columns.
# 6. **train_test_split**: This function splits the dataset into training and testing sets based on a specified test size. It uses random indices to ensure that the split is random and reproducible by setting a seed.
    # why is need to use for both numpy array and pandas DataFrame? Because the input data can be in either format, this function handles both cases to ensure flexibility in data handling.
    # It also checks the type of `X` and `y` to ensure they are either numpy arrays or pandas DataFrames/Series, raising a TypeError if not.
    # If not, it can loss the column name and index of the data frame, which is important for data analysis and model training.
    # A NumPy array requires uniform data types (e.g., all floats), which complicates preprocessing
# 7. **load_data**: This function loads the dataset from a CSV file, preprocesses it (encoding and standardization), and separates the features and target variables. It returns the preprocessed features, target variables, feature names, and statistics.