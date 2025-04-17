from utils import preprocess, evaluate
from models import linear, ann, logistic
import pandas as pd
import numpy as np



data_frame = pd.read_csv('data/heart_attack_dataset.csv')
#print(data_frame)

# Preprocess the data
data_after_ecode = preprocess.encode(data_frame)
#print(data_after_ecode)
data_after_preprocess, stats = preprocess.full_preprocess(data_frame)
#print(data_after_preprocess)
#print(stats)
#columns_to_inverse = ['Age', 'BMI'] change the None to the columns you want to inverse standardize
#data_after_invere_standardize = preprocess.inverse_standardize(data_after_preprocess, stats,None)
#print(data_after_invere_standardize)

X = data_after_preprocess.drop(columns=['Outcome']) # Drop the 'Outcome' column from the DataFrame
y = data_after_preprocess['Outcome']

#print(X)
#print(y)

# Apply train_test_split
X_train, X_test, y_train, y_test = preprocess.train_test_split(X, y, test_size=0.2, seed=42)
#print(X_train, y_train)
#print(X_test, y_test)
#X_train.to_csv('data/X_train.csv', index=False)
#X_test.to_csv('data/X_test.csv', index=False)
#y_train.to_csv('data/y_train.csv', index=False)
#y_test.to_csv('data/y_test.csv', index=False)
