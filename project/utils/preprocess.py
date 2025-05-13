import pandas as pd
import numpy as np
from models import LinearRegression, LogisticRegression
from config import LINEAR_REG_PARAMS, LOGISTIC_REG_PARAMS

# Function to encode categorical variables
def encode(data_frame): 
    df = data_frame.copy()
    categorical_cols = []
    for col in df.columns:
        unique_values = df[col].nunique()
        if unique_values <= 10:
            categorical_cols.append(col)
    for col in categorical_cols:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes
    return df

# Function to standardize numerical variables ( để chuẩn hoá và lưu lại các thông số khi HUẤN LUYỆN MODEL)
def standardize(data_frame): 
    df = data_frame.copy()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col != 'Outcome' and df[col].nunique() > 2]
    stats = {}
    for col in numerical_cols:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std
        stats[col] = {'mean': mean, 'std': std}
    return df, stats


# Function to standardize new data using the statistics from training data ( dùng để chquẩn hoá dữ liệu mới dựa trên các thông số đã lưu lại từ quá trình standardize)
def standardize_with_stats(data_frame, stats):
    df = data_frame.copy()
    for col in stats:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            mean = stats[col]['mean']
            std = stats[col]['std']
            if std != 0:
                df[col] = (df[col] - mean) / std
            else:
                df[col] = 0
    return df

#Mean = 1/n * sum(xi)
#Standard Deviation = sqrt(1/n * sum((xi - Mean)^2))

# Function to inverse standardize ( dùng để đưa dữ liệu về dạng ban đầu)
def inverse_standardize(df, stats, columns=None): 
    df = df.copy()
    if columns is None:
        columns = df.columns.tolist()
    for col in columns:
        if col in stats:
            mean = stats[col]['mean']
            std = stats[col]['std']
            if std != 0:
                df[col] = df[col] * std + mean
            else:
                df[col] = mean
    return df

# Function to preprocess the entire dataset ( kết hợp các bước chuẩn hoá và mã hoá)
def full_preprocess(data_frame):
    df_encoded = encode(data_frame)
    df_standardized, stats = standardize(df_encoded)
    return df_standardized, stats

# Function to split data into training and testing sets ( chia dữ liệu thành 2 phần train và test)
def train_test_split(X, y, test_size=0.2, seed=50):
    np.random.seed(seed)
    if isinstance(X, np.ndarray):
        indices = np.random.permutation(len(X))
    elif isinstance(X, pd.DataFrame):
        indices = np.random.permutation(X.shape[0])
    else:
        raise TypeError("X must be either a numpy array or a pandas DataFrame")
    split_idx = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    if isinstance(X, np.ndarray):
        X_train, X_test = X[train_idx], X[test_idx]
    elif isinstance(X, pd.DataFrame):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    if isinstance(y, np.ndarray):
        y_train, y_test = y[train_idx], y[test_idx]
    elif isinstance(y, pd.Series):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    return X_train, X_test, y_train, y_test

#isinstance(obj, type) checks if obj is an instance of type
# np.random.permutation(n) tạo ra một mảng chứa các số từ 0 đến n-1 được xáo trộn ngẫu nhiên.
# indices[:split_idx] (80% phần tử đầu tiên) là chỉ số cho tập huấn luyện, còn indices[split_idx:] (20% phần tử còn lại) là chỉ số cho tập kiểm tra.


def load_data(data_path):
    data = pd.read_csv(data_path)
    data, stats = full_preprocess(data)  # Assuming this function handles the preprocessing
    
    # Features for Linear Regression (19 features) and Logistic Regression (31 features)
    
    # Features for Linear Regression Model
    linear_features = [
    'Age', 'Gender', 'Ethnicity', 'Income', 'EducationLevel', 'Residence', 'EmploymentStatus', 
    'MaritalStatus', 'Smoker', 'PhysicalActivity', 'AlcoholConsumption', 'Diet', 'StressLevel', 
    'Diabetes', 'Hypertension', 'FamilyHistory', 'Medication', 'PreviousHeartAttack', 'StrokeHistory'
    ]

    # Features for Logistic Regression Model
    logistic_features = linear_features + [
    'Cholesterol', 'BloodPressure', 'HeartRate', 'BMI','MaxHeartRate','ST_Depression','NumberOfMajorVessels', 
    'ChestPainType','ECGResults','ExerciseInducedAngina','Slope', 'Thalassemia'
    ]

    
    # Targets
    targets = ['Cholesterol', 'BloodPressure', 'HeartRate', 'BMI','MaxHeartRate','ST_Depression','NumberOfMajorVessels','ChestPainType','ECGResults','ExerciseInducedAngina','Slope', 'Thalassemia', 'Outcome']

    # Prepare data for Linear Regression model
    X_linear = data[linear_features]
    y_linear = {target: data[target] for target in targets[:12]}  # First 12 targets for linear regression

    # Prepare data for Logistic Regression model
    X_logistic = data[logistic_features]
    y_logistic = {'Outcome': data['Outcome']}  # Only 'Outcome' is the target for logistic regression

    return X_linear, y_linear, X_logistic, y_logistic, linear_features, logistic_features, stats


def load_models():
    # Loading Linear Regression model
    trajectory = LinearRegression(
        lambda_l1=LINEAR_REG_PARAMS['lambda_l1'],
        lambda_l2=LINEAR_REG_PARAMS['lambda_l2'],
        learning_rate=LINEAR_REG_PARAMS['learning_rate'],
        max_iter=LINEAR_REG_PARAMS['max_iter']
    )
    
    # Loading Logistic Regression model
    logistic = LogisticRegression(
        learning_rate=LOGISTIC_REG_PARAMS['learning_rate'],
        max_iter=LOGISTIC_REG_PARAMS['max_iter'],
        lambda_l1=LOGISTIC_REG_PARAMS['lambda_l1'],
        lambda_l2=LOGISTIC_REG_PARAMS['lambda_l2']
    )
    
    return {'Trajectory': trajectory, 'Logistic': logistic}
