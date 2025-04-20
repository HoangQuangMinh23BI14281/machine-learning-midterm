# Machine learning group project Midterm

## Our group 17 ICT Class 1:
| Name| Student ID | Roles |
|:-----------------:|:-----------------:|:-----------------:|
|Hoang Quang Minh|23BI14281|Group leader|
|Cao Hoang Linh|23BI14254|Group member|
|Dang Duc Luong|23BI14273|Group member|
|Tran Hai Minh|23BI14277|Group member|
|Pham The Minh Minh|23BI14279|Group member|

> **Note:** Specific roles will be updated soon

## Project targets:

The main things of this:
1. Take a look through "explain.txt" on the data folder to understand the input

2. Implementation process: <span style="color:red">(ONLY USED "PANDAS", "NUMPY" AND "MATHPLOT"; DONT USED SCIKIT-LEARN)<span>

- Step 1: Data preprocessing
    + Encoding: String data columns (such as gender, ethnicity, employment status) will be encoded into numeric values ​​so that the model can process them.

    + Standardization: Numerical data such as cholesterol, blood pressure, BMI will be standardized to ensure that the values ​​have the same range and units.

    + Handling missing values: Missing values ​​in the data will be handled using methods such as filling in the average value, simulating, or removing rows/columns with serious missing data.

- Step 2: Building a Predictive Model
    
    + Linear Regression Model: To predict indicators such as Cholesterol, Blood Pressure, Heart Rate, BMI. 

    + Logistic Regression Model: To predict the likelihood of having a heart attack (cardiovascular disease). This is a classification problem with two classes: "Yes" and "No". 

    + Deep Learning Model (ANN): An Artificial Neural Network can be used to combine all the factors and generate more accurate predictions.

- Step 3: Prediction and Evaluation
    
    + Cholesterol, BloodPressure, HeartRate, BMI Prediction: The model will predict these indicators within a reasonable range of values ​​based on user input.

    + Heart Attack Risk Prediction: Based on medical and lifestyle factors, the model will indicate the probability that the user has cardiovascular disease or is at risk of having a heart attack.

- Step 4: Diet and Lifestyle Recommendations

    + Dietary Recommendations: Based on indicators such as Cholesterol, BloodPressure, BMI, Diabetes, the model can make suggestions for appropriate diets (e.g., less salt, less sugar, more fruits and vegetables, etc.).

    + Physical Activity Recommendations: If the user has a high BMI or is less physically active, the model can suggest light exercises such as walking or yoga.

    + Risk Warnings: If the user has multiple risk factors (such as smoking, high blood pressure, high cholesterol, etc.), the model will give warnings and encourage them to change their lifestyle or see a doctor.

- Step 5: “What-If” feature
    + Users can try changing certain lifestyle factors (diet, physical activity level, smoking, etc.) and see how the model re-predicts their health indicators and risk of heart attack after changing these factors. 


3. Technology and Tools Used

- Machine Learning: Algorithms such as Linear Regression, Logistic Regression, and ANN (Artificial Neural Networks) will be used to build predictive models.

- Data Preprocessing: Using tools and libraries such as Pandas, NumPy to preprocess data, remove missing data, and normalize metrics.

- Streamlit: Using Streamlit to build a simple user interface for web applications that makes it easy for users to enter data and get prediction results.

Here is the project structure:
<pre>
project/
├── data/
│   ├── heart_attack_data.csv (data_frame we gonna use)
|   ├── explain.txt (explain the meaning of each input)
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
<pre>