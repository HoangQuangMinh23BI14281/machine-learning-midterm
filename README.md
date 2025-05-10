# Machine learning group project Midterm

## Our group 17 ICT Class 1

| Name| Student ID | Roles |
|:-----------------:|:-----------------:|:-----------------:|
|Hoang Quang Minh|23BI14281|Group leader|
|Cao Hoang Linh|23BI14254|Group member|
|Dang Duc Luong|23BI14273|Group member|
|Tran Hai Minh|23BI14277|Group member|
|Pham The Minh |23BI14279|Group member|

## Project overview

Library usages: `PANDAS`, `NUMPY`, `STREAMLIT`

This project uses machine learning to predict health-related features and their long-term evolution, focusing on heart attack risk.

## Project Features

### Part 1: Linear Regression for Lifetime Risk Trajectory

A Linear Regression model predicts the 100-year trajectory of all features based on the ANN-predicted inputs.
Simulates changes in specified features (Income, EducationLevel, Residence, EmploymentStatus, MaritalStatus, PhysicalActivity, AlcoholConsumption, Diet, Smoker, Medication) at random years and tracks their impact on all features.
Uses gradient descent with L1 and L2 regularization.

### Part 2: Logistic Regression for Risk and Feature Change Analysis

A Logistic Regression model calculates the percentage change in heart attack probability at each simulated change point in the trajectory.
Also computes percentage changes for Cholesterol, BloodPressure, BMI, HeartRate, and StressLevel at the same change points.
Uses gradient descent with L1 and L2 regularization.

**Dynamic Regularization:**
Regularization parameters (lambda_l1, lambda_l2) are tuned dynamically using cross-validation for all models.

**Preprocessing:**
Outlier handling using the IQR method to clip extreme values.
Cross-validation for hyperparameter tuning.

**Visualization:**
Trajectories of all 31 features over 100 years, grouped into categories (Demographic, Lifestyle, Medical History, Clinical Tests, Symptoms & Diagnostics).
ROC curve to evaluate the Logistic Regression model's performance.

### Part 3: Evaluation

Evaluation of both models using metrics such as accuracy, precision, recall, F1 score, and log loss for the Logistic Regression model.

## Quick overview for all files

- `main.py`: Main file to run the Streamlit app.
- `train.py`: File to train the models.
- `config.py`: Configuration file for the project.
- `models/linear.py`: Linear regression model.
- `models/logistic.py`: Logistic regression model.
- `utils/evaluate.py`: Evaluation metrics for the models.
- `utils/preprocess.py`: Preprocessing functions for the data.

## Project structure

```makefile
project/
├── data/
│   ├── heart_attack_data.csv (data_frame we gonna use)
│   ├── explain.txt (explain the meaning of each input)
│   ├── linear_weight.npy (Trained data for weights of linear regression model)
│   ├── logistic_weight.npy (Trained data for weights of logistic regression model)
│
├── models/
│   ├── __pycache__/
│   ├── __init__.py (to mark a directory as a Python package)
│   ├── ann.py (Artificial Neural Network)
│   ├── linear.py (linear regression)
│   ├── logistic.py (logistic regression)
├── utils/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── evaluate.py (using to plot the graph)
│   ├── preprocess.py (using to encode the data_frame)
│   ├── config.py (Separate the configuration part from the main code for easy management, maintenance and updating.)
├── main.py
├── train.py
├── .gitignore
├── LICENSE
├── README.md
```

## Running code

**Install dependencies:**

```bash
pip install streamlit pandas numpy matplotlib
```

**Train models (run once):**

```bash
python train.py
```

To train specific models, you can comment out the code for the model you don't want to train.
The models will be placed in the `./project/data/` directory as `linear_weight.npy` and `logistic_weight.npy`.
**Run app:**

```bash
streamlit run main.py
```

To stop the program, press `Ctrl + C` in the terminal.

### Output

#### User inputs

The code will automatically enter inputs from `main.py` including:

- Age(19)
- Gender(Male)
- Ethnicity(Asian)
- Income(975000)
- EducationLevel(College)
- Residence(Urban)
- EmploymentStatus(Employed)
- MaritalStatus(Single)
- Smoker(No)
- PhysicalActivity(0)
- AlcoholConsumption(0)
- Diet(Unhealthy)
- StressLevel(6.0)
- Diabetes(Yes)
- Hypertension(Yes)
- FamilyHistory(Yes)
- Medication(No)
- PreviousHeartAttack(Yes)
- StrokeHistory(Yes)

#### Prediction

- Cholesterol
- BloodPressure
- BMI
- MaxheartRate
- ST_depression
- ChestPainType
- ECGResults
- ExerciseInducedAngina
- Slope
- Thallassmia

And will calculate the Heart Attack Probability based on the predicted features and user inputs.

#### Model Evaluation

- For the Logistic Regression model, the code will calculate the following: Accuracy, Precision, Recall, F1 Score, and Log Loss.
- For the Linear Regression model, the code will calculate the following: Mean Absolute Error, Mean Squared Error, Root Mean Squared Error, and R2 Score.
