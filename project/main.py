import numpy as np
import pandas as pd
from colorama import init, Fore, Back, Style  # For colored terminal output

from utils import load_data, load_models, standardize_with_stats, train_test_split, inverse_standardize
from utils.evaluate import accuracy, precision, recall, f1_score, log_loss, mae, mse, r2_score

def run_model():
    # Initialize colorama
    init(autoreset=True)
    
    # Load data and models
    print(f"{Fore.CYAN}{Style.BRIGHT}Loading data and models...")
    X_linear, y_linear, X_logistic, y_logistic, linear_features, logistic_features, stats = load_data('./project/data/heart_disease_data.csv')

    # Split data for Logistic Regression model
    X_train_logistic, X_test_logistic, y_train_outcome, y_test_outcome = train_test_split(X_logistic[logistic_features], y_logistic['Outcome'], test_size=0.2, seed=50)
    # Split data for Linear Regression model (if needed for other parts of the project)
    X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(X_linear[linear_features], y_linear['Cholesterol'], test_size=0.2, seed=50)
    # Load models
    models = load_models()

    # Load pre-trained weights
    print(f"{Fore.YELLOW}Loading pre-trained weights for models...")
    try:
        models['Trajectory'].load_weights('./project/data/linear_weights.npy')
        print(f"{Fore.GREEN}Linear Regression weights loaded successfully.")
    except Exception as e:
        print(f"{Fore.RED}Failed to load Linear Regression weights: {str(e)}")

    try:
        models['Logistic'].load_weights('./project/data/logistic_weights.npy')
        print(f"{Fore.GREEN}Logistic Regression weights loaded successfully.")
    except Exception as e:
        print(f"{Fore.RED}Failed to load Logistic Regression weights: {str(e)}")

    # Define feature categories (use separate categories for linear and logistic features)
    feature_groups = {
        "Demographic": ['Age', 'Gender', 'Ethnicity', 'Income', 'EducationLevel', 'Residence', 'EmploymentStatus', 'MaritalStatus'],
        "Lifestyle": ['Smoker', 'PhysicalActivity', 'AlcoholConsumption', 'Diet', 'StressLevel'],
        "Medical History": ['Diabetes', 'Hypertension', 'FamilyHistory', 'Medication', 'PreviousHeartAttack', 'StrokeHistory'],
        
        "Clinical Tests": ['Cholesterol', 'BloodPressure', 'HeartRate', 'BMI', 'MaxHeartRate', 'ST_Depression', 'NumberOfMajorVessels'],
        "Symptoms & Diagnostics": ['ChestPainType', 'ECGResults', 'ExerciseInducedAngina', 'Slope', 'Thalassemia']
    }

    # Required features for demographic, lifestyle, medical history, and predictive features
    required_features = feature_groups["Demographic"] + feature_groups["Lifestyle"] + feature_groups["Medical History"]
    predict_features = feature_groups["Clinical Tests"] + feature_groups["Symptoms & Diagnostics"]
    integer_features = ['Age', 'StressLevel', 'PhysicalActivity', 'AlcoholConsumption', 'NumberOfMajorVessels']
    decimal_features = ['Cholesterol', 'BloodPressure', 'HeartRate', 'BMI', 'MaxHeartRate', 'ST_Depression', 'Income']
    
    # Load raw data to compute default values
    data_raw = pd.read_csv('./project/data/heart_disease_data.csv')
    default_values = {}
    for feature in required_features + predict_features:
        if feature in ['Age', 'Income', 'StressLevel', 'MaxHeartRate', 'ST_Depression', 'NumberOfMajorVessels', 'Cholesterol', 'BloodPressure', 'HeartRate', 'BMI']:
            default_values[feature] = data_raw[feature].median()
        else:
            default_values[feature] = data_raw[feature].mode()[0] if not data_raw[feature].mode().empty else 0

    # Mapping for categorical features (unchanged)
    feature_mappings = {
        'Gender': {'Male': 1, 'Female': 0},
        'Diet': {'Unhealthy': 0, 'Healthy': 1, 'Moderate': 2},
        'Ethnicity': {'Hispanic': 0, 'Asian': 1, 'Black': 2, 'Other': 3, 'White': 4},
        'EducationLevel': {'High School': 0, 'College': 1, 'Postgraduate': 2},
        'Medication': {'Yes': 1, 'No': 0},
        'ChestPainType': {'Typical': 0, 'Atypical': 1, 'Non-anginal': 2, 'Asymptomatic': 3},
        'ECGResults': {'ST-T abnormality': 0, 'LV hypertrophy': 1, 'Normal': 2},
        'ExerciseInducedAngina': {'Yes': 1, 'No': 0},
        'Slope': {'Downsloping': 0, 'Upsloping': 1, 'Flat': 2},
        'Thalassemia': {'Normal': 0, 'Reversible defect': 1, 'Fixed defect': 2},
        'Residence': {'Suburban': 0, 'Rural': 1, 'Urban': 2},
        'EmploymentStatus': {'Retired': 0, 'Unemployed': 1, 'Employed': 2},
        'MaritalStatus': {'Single': 0, 'Married': 1, 'Widowed': 2, 'Divorced': 3},
        'Smoker': {'Yes': 1, 'No': 0},
        'Diabetes': {'Yes': 1, 'No': 0},
        'Hypertension': {'Yes': 1, 'No': 0},
        'FamilyHistory': {'Yes': 1, 'No': 0},
        'PreviousHeartAttack': {'Yes': 1, 'No': 0},
        'StrokeHistory': {'Yes': 1, 'No': 0},
        'PhysicalActivity': {str(i): i for i in range(7)},
        'AlcoholConsumption': {str(i): i for i in range(5)}
    }

    # Simulate user inputs (use correct feature set based on what we need)
    user_inputs = {feature: default_values[feature] for feature in required_features}
    user_inputs.update({
        'Age': 70,
        'Gender': feature_mappings['Gender']['Male'],
        'Ethnicity': feature_mappings['Ethnicity']['Asian'],
        'Income': 97500.0,
        'EducationLevel': feature_mappings['EducationLevel']['College'],
        'Residence': feature_mappings['Residence']['Urban'],
        'EmploymentStatus': feature_mappings['EmploymentStatus']['Employed'],
        'MaritalStatus': feature_mappings['MaritalStatus']['Single'],
        'Smoker': feature_mappings['Smoker']['No'],
        'PhysicalActivity': 0,
        'AlcoholConsumption': 0,
        'Diet': feature_mappings['Diet']['Unhealthy'],
        'StressLevel': 6,
        'Diabetes': feature_mappings['Diabetes']['Yes'],
        'Hypertension': feature_mappings['Hypertension']['Yes'],
        'FamilyHistory': feature_mappings['FamilyHistory']['Yes'],
        'Medication': feature_mappings['Medication']['No'],
        'PreviousHeartAttack': feature_mappings['PreviousHeartAttack']['Yes'],
        'StrokeHistory': feature_mappings['StrokeHistory']['Yes']
    })

    # Convert user inputs to DataFrame
    input_df = pd.DataFrame([user_inputs], columns=required_features)
    # Standardize input
    input_std = standardize_with_stats(input_df, stats)
    X_input = input_std.values

    # Step 1: Predict and compute probability
    print(f"{Fore.CYAN}{Style.BRIGHT}Predicting clinical tests and symptoms...")
    X_pred = models['Trajectory'].predict(X_input)
    if X_pred.ndim == 1:
        X_pred = X_pred.reshape(1, -1)
    pred_df = pd.DataFrame(X_pred, columns=predict_features)
    # Inverse standardize
    input_original = inverse_standardize(pd.DataFrame(input_std.values, columns=required_features), stats)
    pred_df = inverse_standardize(pred_df, stats)

    # Hàm để tra cứu giá trị trong feature_mappings với kiểm tra khớp
    def map_feature_value(feature, value): 
        # Làm tròn giá trị thành số nguyên cho các cột phân loại
        value = round(value)  # Làm tròn giá trị
        value = int(value)  # Đảm bảo là int
        
        # Kiểm tra nếu feature có trong feature_mappings
        if feature in feature_mappings:
            # Kiểm tra nếu giá trị có trong feature_mappings
            if value in feature_mappings[feature].values():
                mapped_value = [k for k, v in feature_mappings[feature].items() if v == value]
                return mapped_value[0] if mapped_value else 'Unknown'
            else:
                return 'Unknown'
        return value

    for feature in feature_mappings:
        if feature in input_original.columns:
            input_original[feature] = input_original[feature].apply(lambda x: map_feature_value(feature, x))

    # Cập nhật các giá trị cho các cột phân loại
    for feature in required_features + predict_features:
        if feature in pred_df.columns:
            if feature in feature_mappings:  # Xử lý các giá trị dạng chữ
                pred_df[feature] = pred_df[feature].apply(lambda x: map_feature_value(feature, x))
            else:  # Xử lý các giá trị số
                pred_df[feature] = pred_df[feature].apply(lambda x: int(round(x)) if isinstance(x, (float, int)) else x)

    # ------------------ USER INPUTS AND PREDICTIONS ------------------
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*80}")
    print(f"{Fore.CYAN}{Style.BRIGHT}                       USER INPUTS AND PREDICTED FEATURES")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}")
    
    print(f"\n{Fore.GREEN}{Style.BRIGHT}USER INPUTS:")
    for feature in required_features:
        if feature in input_original.columns:
            value = input_original[feature].iloc[0]
            print(f"{Fore.WHITE}{feature}: {Fore.YELLOW}{value}")
    
    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}PREDICTED FEATURES:")
    for feature in predict_features:
        if feature in pred_df.columns:
            value = pred_df[feature].iloc[0]
            print(f"{Fore.WHITE}{feature}: {Fore.YELLOW}{value}")

    user_inputs.update({
        'Cholesterol': 220, 
        'BloodPressure': 120,  
        'HeartRate': 75, 
        'BMI': 25.0,  
        'MaxHeartRate': 180,  
        'ST_Depression': 1.2,  
        'NumberOfMajorVessels': 0,  
        'ChestPainType': feature_mappings['ChestPainType']['Atypical'], 
        'ECGResults': feature_mappings['ECGResults']['Normal'],  
        'ExerciseInducedAngina': feature_mappings['ExerciseInducedAngina']['No'],  
        'Slope': feature_mappings['Slope']['Upsloping'],  
        'Thalassemia': feature_mappings['Thalassemia']['Normal']  
    })

    input_df = pd.DataFrame([user_inputs])
    input_std = standardize_with_stats(input_df, stats)
    X_input = input_std.values

    # Compute heart attack probability
    initial_prob = models['Logistic'].predict_proba(X_input)[0]
    print(f"\n{Fore.RED}{Style.BRIGHT}HEART ATTACK PROBABILITY: {Fore.WHITE}{initial_prob*100:.2f}%")

    # ------------------ MODEL EVALUATION ------------------
    print(f"\n\n{Fore.CYAN}{Style.BRIGHT}{'='*80}")
    print(f"{Fore.CYAN}{Style.BRIGHT}                            MODEL EVALUATION")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}")
    
    # Logistic Regression evaluation with inverse standardization
    X_test_logistic_original = inverse_standardize(pd.DataFrame(X_test_logistic.values, columns=logistic_features), stats)
    X_test_logistic_std = standardize_with_stats(X_test_logistic_original, stats)

    logit_pred_proba = models['Logistic'].predict_proba(X_test_logistic_std.values)
    logit_predictions = (logit_pred_proba >= 0.5).astype(int)  # Positive class threshold
    logit_accuracy = accuracy(y_test_outcome.values, logit_predictions)
    logit_precision = precision(y_test_outcome.values, logit_predictions)
    logit_recall = recall(y_test_outcome.values, logit_predictions)
    logit_f1 = f1_score(y_test_outcome.values, logit_predictions)
    logit_log_loss = log_loss(y_test_outcome.values, logit_pred_proba)

    print(f"\n{Fore.BLUE}{Style.BRIGHT}LOGISTIC REGRESSION METRICS:")
    print(f"{Fore.WHITE}Accuracy:  {Fore.GREEN}{logit_accuracy:.4f}")
    print(f"{Fore.WHITE}Precision: {Fore.GREEN}{logit_precision:.4f}")
    print(f"{Fore.WHITE}Recall:    {Fore.GREEN}{logit_recall:.4f}")
    print(f"{Fore.WHITE}F1-Score:  {Fore.GREEN}{logit_f1:.4f}")
    print(f"{Fore.WHITE}Log Loss:  {Fore.GREEN}{logit_log_loss:.4f}")

    # Linear Regression evaluation with inverse standardization
    linear_predictions = models['Trajectory'].predict(X_test_linear.values)
    linear_predictions = linear_predictions[:, 0]

    # Convert predictions and actual values to DataFrames for inverse standardization
    pred_df = pd.DataFrame(linear_predictions, columns=['Cholesterol'])
    actual_df = pd.DataFrame(y_test_linear.values, columns=['Cholesterol'])

    # Inverse standardize both predictions and actual values
    pred_original = inverse_standardize(pred_df, stats)
    actual_original = inverse_standardize(actual_df, stats)

    # Calculate metrics using inverse standardized values
    linear_mae = mae(actual_original['Cholesterol'].values, pred_original['Cholesterol'].values)
    linear_mse = mse(actual_original['Cholesterol'].values, pred_original['Cholesterol'].values)
    linear_r2 = r2_score(actual_original['Cholesterol'].values, pred_original['Cholesterol'].values)

    # Print results
    print(f"\n{Fore.BLUE}{Style.BRIGHT}LINEAR REGRESSION METRICS:")
    print(f"{Fore.WHITE}MAE:      {Fore.GREEN}{linear_mae:.4f}")
    print(f"{Fore.WHITE}MSE:      {Fore.GREEN}{linear_mse:.4f}")
    print(f"{Fore.WHITE}R² Score: {Fore.GREEN}{linear_r2:.4f}")
    
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*80}")
    print(f"{Fore.CYAN}{Style.BRIGHT}                            END OF REPORT")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}")

# MAIN function with simple terminal UI
def main():
    print(f"{Fore.YELLOW}{Style.BRIGHT}{'='*80}")
    print(f"{Fore.YELLOW}{Style.BRIGHT}                HEART ATTACK RISK PREDICTION MODEL")
    print(f"{Fore.YELLOW}{Style.BRIGHT}{'='*80}")
    print(f"{Fore.CYAN}Running model with default test parameters...")
    run_model()
    

if __name__ == "__main__":
    # Initialize colorama at the very beginning
    init(autoreset=True)
    main()