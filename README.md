 Heart Disease Prediction System

A Machine Learning-based "Heart Disease Prediction System" built using "Python, Streamlit, and Machine Learning algorithms". This project predicts the risk of heart disease using patient health parameters and provides both "single patient prediction" and "bulk CSV prediction" functionality.

 Features

 Predict heart disease risk for an individual patient  
 Upload CSV file for "bulk prediction"  
 Compare results from multiple ML models  
 Interactive "risk percentage gauge chart"  
 Clean and responsive "Streamlit UI"  
 Download prediction results as CSV file  


 Technologies Used

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Plotly
- Pickle

Machine Learning Model Used

- LogisticRegression
- DecisionTree
- RandomForest

Input Features
The model predicts heart disease using the following 11 medical attributes:

Age
Sex
Chest Pain Type
Resting Blood Pressure
Cholesterol
Fasting Blood Sugar
Resting ECG
Maximum Heart Rate
Exercise-Induced Angina
Oldpeak
ST Slope

Application Tabs

1️ Predict
Users can enter patient medical information manually and get:
Heart disease risk percentage
Final prediction (High Risk / Low Risk)
Predictions from all ML models

2️ Bulk Predict
Upload a CSV file with patient data to:
Predict multiple patients at once
Download prediction results as CSV

3️ Model Info 
Displays model accuracy comparison using charts.

Install Dependencies
pip install -r requirements.txt
Run Streamlit App
streamlit run app.py
  
 Project Structure

Heart-Disease-Prediction/
│── app.py
│── Heart_disease_prediction.ipynb
│── LogisticRegression.pkl
│── DecisionTree.pkl
│── RandomForest.pkl
│── requirements.txt
│── README.md


Author
Minahil Tariq
