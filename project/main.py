from utils import preprocess, evaluate
from models import linear, ann, logistic
import pandas as pd
import numpy as np



data_frame = pd.read_csv('data/heart_attack_dataset.csv')
#print(data_frame)

# Preprocess the data
data_after_preprocess, stats = preprocess.full_preprocess(data_frame)
print(data_after_preprocess)
print(stats)
