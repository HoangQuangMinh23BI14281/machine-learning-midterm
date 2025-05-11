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

Library usages: `PANDAS`, `NUMPY`,

Our project is a machine learning system, with the purpose of predicting heart attack risks. We use a custom built model and a dataset.

Focus:

- Predicting clinical indicators
- Estimate risks of the heart attack based on input features
- Evaluating performance of the model

## Project Features

## Part 1: Input features

- The system use 19 input features to predict heart attack:
  - Age
  - Gender
  - Ethnicity
  - Income
  - EducationLevel
  - Residence
  - EmploymentStatus
  - MaritalStatus
  - Smoker
  - PhysicalActivity
  - AlcoholConsumption
  - Diet
  - StressLevel
  - Diabetes
  - Hypertension
  - FamilyHistory
  - Medication
  - PreviousHeartAttack
  - StrokeHistory

### Part 2: Linear Regression

Our linear regression model predicts the following clinical indicators:

- Predict 12 Clinical features:
  - Cholesterol
  - BloodPressure
  - BMI
  - MaxheartRate
  - ST_depression
  - ChestPainType
  - ECGResults
  - ExerciseInducedAngina
  - Slope
  - Thalassemia
- Implement : Gradient Decent with L1 and L2 regularization to control complexity
- Support both L1 and L2 regularization with early stopping
  - L1:Lasso regression helps to driving weights to zero
  - L2:Ridge regression helps to prevent large coefficients
- Track cost history
- Use random initialization
- Use Numpy

### Part 3: Logistic Regression

Our Logistic regression calculate heart attack probability based on the following features:

- Process all features to predict heart attack risks
- Use sigmoid function, batch gradient descent function
- Implement L1 and L2 regularization to control complexity
  - L1:Lasso regression helps to enforce sparsity and feature selection
  - L2:Ridge regression helps  with weight shrinkage and stability
- Use Numpy

### Part 4: Evaluation

We implement a model evaluation to assess prediction quality and performance. The evaluation includes:

- For the Logistic Regression model:
  - Accuracy: Measure the proportion of correct predictions to the total predictions made.
  - Precision: Evaluate model's ability to avoid false positives.
  - Recall: Assess the model to find all positive cases
  - F1 Score: Harmonic mean of precision and recall, providing a balance between the two.
  - Log Loss: Evaluate the uncertanty of predictions
- For the Linear Regression model:
  - Mean Absolute Error(MAE): Average of absolute differences between predicted and actual values.
  - Mean Squared Error(MSE): Average of squared differences between predicted and actual values.
  - R2 Score: Measure of how well the model fits the data.

## File descriptions

- `main.py`: The main file to load pre-trained models, and run the prediction. Also responsible for loading user inputs, running predictions, and displaying results.
- `train.py`: Training script to load data, initialize models with configured parameters, splits training, testing data, fits models then save the trained data to `data/` directory.
- `config.py`: Configuration files containing parameters for both models: Learning rate, maximum iterations, regularization parameters `l1` and `l2`

- `models/linear.py`: Linear regression model containing
  - Gradient descent optimization
  - Cost function
  - `l1` and `l2` regularization
  - Weight management
  - Prediction function
- `models/logistic.py`: Logistic regression model containing
  - Sigmoid function
  - Cost function
  - `l1` and `l2` regularization
  - Weight management
  - Probabilistic output
- `utils/evaluate.py`: Evaluation metrics for the models:
  - Logistic regression: Accuracy, Precision, Recall, F1 Score, Log Loss
  - Linear regression: Mean Absolute Error, Mean Squared Error, Root Mean Squared Error, R2 Score
- `utils/preprocess.py`: Preprocessing functions for the data:
  - Feature encoding,standardization
  - Train splitting function
  - Data loading function, transformation utilities
- `__init__.py`: Package initialization files, having key functions from utils modules.

## Project structure

```makefile
project/
├── data/
│   ├── heart_disease_data.csv (data_frame we gonna use)
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

## Implementation

### Data preprocessing

- Feature encoding: Categorical variables are encoded to numerical values
- Standardization: Features are scaled to have a mean of 0 and a unit variance

### Model training

- Gradient decent: Implemented with configurable rates
- Regularization: L1 and L2 regularization to control model complexity and better generalization
- Early stopping to prevent overfitting

### Evaluation

- Cross validation: Use for hyperparamenter tuning, model selection
- Metrics tracking: Capture different aspects of the performance
- Standardized reporting: Use for easy interpretation

## Running code

**Install dependencies:**

```bash
pip install pandas numpy matplotlib 
```

**Train models (run once):**

```bash
python train.py
```

The models will be placed in the `./project/data/` directory as `linear_weight.npy` and `logistic_weight.npy`.
**Run application:**

```bash
python main.py
```

To stop the program, press `Ctrl + C` in the terminal.

## Usage

- The application will load the pre-trained models, and the dataset
- User input are provided automatically and can be customized
- The Linear regression model will predict the clinical indicators
- The Logistic regression model will predict the heart attack risks
- The evaluation metrics will be displayed for both models
