from utils import preprocess, evaluate
from models import logistic, ann
import pandas as pd
import numpy as np



data_frame = pd.read_csv('data/heart_attack_dataset.csv')
#print(data_frame)
categorical_cols = []
for col in data_frame.columns:
    unique_values = data_frame[col].nunique()
    #print(f"{col}, Unique Values: {unique_values}")
#print(data_frame.dtypes)

X, y, features, stats = preprocess.load_data('data/heart_attack_dataset.csv')
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
