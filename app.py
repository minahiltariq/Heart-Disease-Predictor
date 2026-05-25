import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>.main {background-color: #f5f7fa;}

.big-title {
    font-size: 40px;
    font-weight: bold;
    color: #d62828;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 20px;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown("<p class='big-title'>🫀 Heart Disease Predictor</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict Heart Disease Risk using Machine Learning</p>",unsafe_allow_html=True)


# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📂 Bulk Predict", "📊 Model Info"])

# ============================================================
# TAB 1
# ============================================================
with tab1:

    st.markdown("### Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 150)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type",["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"])
        resting_bp = st.number_input("Resting Blood Pressure", 0, 300)
        cholesterol = st.number_input("Cholesterol", 0)
        fasting_bs = st.selectbox("Fasting Blood Sugar",["<=120 mg/dl", ">120 mg/dl"])

    with col2:
        resting_ecg = st.selectbox("Resting ECG",["Normal","ST-T Wave Abnormality","Left Ventricular Hypertrophy"])
        max_hr = st.number_input("Maximum Heart Rate",60, 202)
        exercise_angina = st.selectbox("Exercise Angina",["Yes", "No"])
        oldpeak = st.number_input("Oldpeak",0.0, 10.0)

        st_slope = st.selectbox("ST Slope",["Upsloping", "Flat", "Downsloping"])

    # ---------------- Mapping ----------------
    sex = 0 if sex == "Male" else 1
    chest_pain = ["Atypical Angina","Non-Anginal Pain","Asymptomatic","Typical Angina"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == ">120 mg/dl" else 0
    resting_ecg = ["Normal","ST-T Wave Abnormality","Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping","Flat","Downsloping"].index(st_slope)

# Create a DataFrame with user inputs
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    modelnames = ['DecisionTree.pkl','LogisticRegression.pkl','RandomForest.pkl']
    algonames = ['Decision Tree','Logistic Regression','Random Forest']

    def predict_heart_disease(data):
        probs = []

        for modelname in modelnames:
            model = pickle.load(open(modelname, 'rb'))
            prob = model.predict_proba(data)[0][1] # probability of class 1 (heart disease)
            probs.append(prob * 100) # convert to percentage

        return probs

    if st.button("🩺 Analyse Risk", use_container_width=True):
        results = predict_heart_disease(input_data)
        avg_probability = round(np.mean(results), 2)

        final_prediction = ("High Risk 🚨"
            if avg_probability >= 50
            else "Low Risk ✅"
        )
        st.markdown("---")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Final Prediction")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_probability,
                title={'text': "Heart Disease Risk %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red"},
                    'steps': [{'range': [0, 30],'color': "lightgreen"},{'range': [30, 70],'color': "yellow"},{'range': [70, 100],'color': "salmon"}]
                }
            ))

            fig.update_layout(height=350)
            st.plotly_chart(fig,use_container_width=True)
            st.success(final_prediction)

        with col2:
            st.subheader("Model Predictions")

            for i in range(len(results)):
                st.metric(algonames[i],f"{results[i]:.2f}%")

# ============================================================
# TAB 2
# ============================================================
with tab2:

    st.subheader("📂 Upload CSV File")

    st.subheader('Instructions to note before uploading the file:')
    st.info("""
    1. No NaN values allowed.
    2. Total 11 features in this order ('Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope').\n
    3. Check the spellings of the feature names.
    4. Feature values conventions: \n
       - Age: age of the patient [years] \n
       - Sex: sex of the patient [0: Male, 1: Female] \n
       - ChestPainType: chest pain type [3: Typical Angina, 0: Atypical Angina, 1: Non-Anginal Pain, 2: Asymptomatic] \n
       - RestingBP: resting blood pressure [mm Hg] \n
       - Cholesterol: serum cholesterol [mm/dl] \n
       - FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise] \n
       - RestingECG: resting electrocardiogram results [0: Normal, 1: having ST-T wave abnormality (T wave inversions and/or ST deviation), ...]\n
       - MaxHR: maximum heart rate achieved [Numeric value between 60 and 202] \n
       - ExerciseAngina: exercise-induced angina [1: Yes, 0: No] \n
       - Oldpeak: oldpeak = ST [Numeric value measured in depression] \n
       - ST_Slope: the slope of the peak exercise ST segment [0: upsloping, 1: flat, 2: downsloping] \n
    """)

    uploaded_file = st.file_uploader("Upload CSV",type=["csv"])

    if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
        input_data = pd.read_csv(uploaded_file)
        model = pickle.load(open('LogisticRegression.pkl', 'rb'))

        predictions = model.predict(input_data)

        input_data["Prediction"] = predictions
        st.success("Prediction Completed!")
        st.dataframe(input_data)
        csv = input_data.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Predictions",csv,"predictions.csv","text/csv")

# ============================================================
# TAB 3
# ============================================================
with tab3:

    st.subheader("Model Accuracy")
    data = {'Decision Tree': 80.97,'Logistic Regression': 85.86,'Random Forest': 88.54}
    df = pd.DataFrame(list(data.items()),columns=['Model', 'Accuracy'])
    st.bar_chart(df.set_index('Model'))