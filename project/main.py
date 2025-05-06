from utils import preprocess, evaluate
from models import logistic, ann
import pandas as pd
import numpy as np



data_frame = pd.read_csv('./project/data/heart_attack_dataset.csv') # Load the dataset, change the path if needed
#print(data_frame)
categorical_cols = []
for col in data_frame.columns:
    unique_values = data_frame[col].nunique()
    print(f"{col}, Unique Values: {unique_values}")
print(data_frame.dtypes)
# Preprocess the data
data_after_preprocess, stats = preprocess.full_preprocess(data_frame)
#print(data_after_preprocess)
#print(stats)
#columns_to_inverse = ['Age', 'BMI'] change the None to the columns you want to inverse standardize
#data_after_invere_standardize = preprocess.inverse_standardize(data_after_preprocess, stats,None)
#print(data_after_invere_standardize)

#X = data_after_preprocess.drop(columns=['Outcome']) # Drop the 'Outcome' column from the DataFrame
#y = data_after_preprocess['Outcome']
 
#print(X)
#print(y)

X, y, features, stats = preprocess.load_data('./project/data/heart_attack_dataset.csv') # Load the dataset, change the path if needed
#print(X)
#print(y)
#print(features)
#print(stats)

# Apply train_test_split
#X_train, X_test, y_train, y_test = preprocess.train_test_split(X, y['Outcome'], test_size=0.2, seed=42)
#print(X_train, y_train)
#print(X_test, y_test)
#X_train.to_csv('data/X_train.csv', index=False) # Becareful with this line, it will make a csv file too large to be push on github
#X_test.to_csv('data/X_test.csv', index=False)
#y_train.to_csv('data/y_train.csv', index=False)
#y_test.to_csv('data/y_test.csv', index=False)
