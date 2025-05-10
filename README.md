# Machine learning group project Midterm

## Our group 17 ICT Class 1

| Name| Student ID | Roles |
|:-----------------:|:-----------------:|:-----------------:|
|Hoang Quang Minh|23BI14281|Group leader|
|Cao Hoang Linh|23BI14254|Group member|
|Dang Duc Luong|23BI14273|Group member|
|Tran Hai Minh|23BI14277|Group member|
|Pham The Minh |23BI14279|Group member|

## Project Health Prediction System

> (ONLY USE `PANDAS`, `NUMPY`, `STEAMLIT` AND `MATHPLOT`; DONT USE `SCIKIT-LEARN`)

This project uses machine learning to predict health-related features and their long-term evolution, focusing on heart attack risk.

### Features

#### Part 1: ANN for Feature Prediction

An Artificial Neural Network (ANN) predicts all 31 features based on a subset of user-provided inputs. As more inputs are provided, predictions are refined.
Uses gradient descent with backpropagation, L1 and L2 regularization.

#### Part 2: Linear Regression for Lifetime Risk Trajectory

A Linear Regression model predicts the 100-year trajectory of all features based on the ANN-predicted inputs.
Simulates changes in specified features (Income, EducationLevel, Residence, EmploymentStatus, MaritalStatus, PhysicalActivity, AlcoholConsumption, Diet, Smoker, Medication) at random years and tracks their impact on all features.
Uses gradient descent with L1 and L2 regularization.

#### Part 3: Logistic Regression for Risk and Feature Change Analysis

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

### Model Performance

ROC Curve: A plot showing the ROC curve with an AUC (e.g., 0.85), a diagonal line for reference, and labels for FPR and TPR.

### Considerations

#### Trajectory Visualization

Features are grouped into categories to make the 100-year trajectories more manageable and interpretable.
Each plot includes change points marked with vertical dashed lines for clarity.

#### Risk and Feature Change Analysis

Percentage changes are computed for Heart Attack Probability, Cholesterol, BloodPressure, BMI, HeartRate, and StressLevel at each simulated change point.
These changes help understand the impact of simulated interventions on both risk and health metrics.

#### ROC Curve

The ROC curve is computed using the test set, providing a static evaluation of the Logistic Regression model’s performance.
The AUC value gives users a sense of the model’s reliability (e.g., AUC > 0.8 indicates good performance).

#### Dynamic Regularization

Regularization parameters are tuned using cross-validation, which improves model generalization but increases training time.

#### Performance

The app remains responsive with real-time updates.
Dynamic regularization and cross-validation may slow down the training process; consider reducing the parameter grid or number of folds if performance is a concern.

#### Limitations

For informational purposes only, not medical advice.
Model accuracy depends on dataset quality.
The 100-year trajectory is a simplified projection and does not account for complex real-world factors.
Training time may increase due to dynamic regularization and cross-validation.

### Project structure

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

### Setup

**Install dependencies:**

```bash
pip install streamlit pandas numpy matplotlib
```

**Train models (run once):**

```bash
python train.py
```

The models will be placed in the `./project/data/` directory as `linear_weight.npy` and `logistic_weight.npy`.
**Run app:**

```bash
streamlit run main.py
```

### Usage

*Step 1: Predict Remaining Features:*
Enter one or more features (e.g., Age, Gender, Smoker).
The ANN predicts the remaining features, updating predictions as you provide more inputs.

*Step 2: 100-Year Lifetime Risk Trajectory:*
View the 100-year trajectory of all 31 features, grouped into categories.
The app simulates changes in specified features at random years and displays their impact on the trajectories.

*Step 3: Risk and Feature Change Analysis:*
View the percentage change in Heart Attack Probability, Cholesterol, BloodPressure, BMI, HeartRate, and StressLevel at each simulated change point in the trajectory.
Evaluate the Logistic Regression model's performance with an ROC curve.

### Output
