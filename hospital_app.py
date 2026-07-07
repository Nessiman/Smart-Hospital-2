import streamlit as st
import pandas as pd
import pickle

# untuk membuka model
with open("hospital_model.pkl","rb") as f:
  bundle = pickle.load(f)
  st.write("Connected")

# untuk import model
model = bundle["model"]
scaler = bundle["scaler"]

# memanggil features
features = bundle["features"]
cols_to_scale = bundle["cols_to_scale"]

dept_map_inv = bundle["dept_map_inv"]

gender_map = bundle["gender_map"]
temp_map = bundle["temp_map"]
hr_map = bundle["hr_map"]
dur_map = bundle["dur_map"]
cc_map = bundle["cc_map"]

DEPT_INFO = {
  "Respiratory Medicine": {
    "icon" : "🫁",
    "desc" : "Specialises in condition affexting the lungs and airways",
    "next" : [
      "visit level 2, Wing B",
      "Estimated wait: 15-25 minutes",
      "Please wear mask"
    ]
  },
  "Cardiology": {
    "icon" : "💗",
    "desc" : "Specialises in heart and cardiovascular conditions",
    "next" : [
      "visit level 3, Wing A",
      "Estimated wait: 20-30 minutes",
      "Please bring previous ECG reports"
    ]
  },
  "Gastroenterology": {
    "icon" : "🫄",
    "desc" : "Specialises in digestive system condition",
    "next" : [
      "visit level 1, Wing C",
      "Estimated wait: 20-30 minutes",
    ]
  },
"Neurology": {
    "icon" : "🧠",
    "desc" : "Specialises in brain and nervous system condition",
    "next" : [
      "visit level 4, Wing A",
      "Estimated wait: 30 - 60 minutes",
      "Bring current medicatoin list"
    ]
  },
"General Medicine": {
    "icon" : "🩺",
    "desc" : "Genereal Health Consultation",
    "next" : [
      "Visit level 1, Wing A"
    ]
  },

"Dermatology": {
    "icon" : "🧏🏼‍♀️",
    "desc" : "Specialises in skin conditions",
    "next" : [
      "Visit level 2, Wing D"
    ]
  }
}

# Creating The dasboard
st.title("🏥 Smart Hospital Patient Navigator Ms Annes")

st.write("Fill in the patient's information below")

st.header("Patient Information")

age = st.number_input(
  "Age",
  min_value=1,
  max_value=120,
  value=30
)
gender = st.selectbox(
  "Gender",
  ["Female", "Male"]
)

# symptomps
st.header("Symptomps")

col1, col2 = st.columns(2)

with col1:
  fever = st.checkbox("Fever 🥵")
  cough = st.checkbox("Cough 😷")
  headache = st.checkbox("Headache 💆‍♂️")
  chest_pain = st.checkbox("Chest Pain💔")
  stomach_pain = st.checkbox("Stomach Pain 🤢")

with col2:
  shortness_breath = st.checkbox("Shortness Breath 😮‍💨")
  nausea_vomiting = st.checkbox("Nausea/Vomiting 🤮")
  dizziness = st.checkbox("Dizziness 😵‍💫")
  skin_rash = st.checkbox("Skin Rash 🔴")

st.header("Patient Condition")
temperature_level = st.selectbox(
  "Temperature",
  options=list(temp_map.keys())
)

heart_rate_level = st.selectbox(
  "Heart Rate",
  options=list(hr_map.keys())
)

duration = st.selectbox(
  "Duration of Symptoms",
  options=list(dur_map.keys())
)
chief_complaint = st.selectbox(
  "Chieft Complaint",
  options=list(cc_map.keys())
)

st.header("Medical History")
hypertension = st.checkbox("Hypertension")

heart_disease = st.checkbox("Heart Disease")

asthma = st.checkbox("Asthma")

# Prediction section

predict_button = st.button("Predict Department")

if predict_button: 
  patient =pd.DataFrame([{
    "age" : age,
    "gender" : gender_map[gender],
    "fever" : int(fever),
    "cough" : int(cough),
    "headache" : int(headache),
    "chest_pain" : int(chest_pain),
    "stomach_pain" : int(stomach_pain),
    "shortness_breath" : int(shortness_breath),
    "nausea_vomiting" : int(nausea_vomiting),
    "dizziness" : int(dizziness),
    "skin_rash" : int(skin_rash),

    "temperature_level" : temp_map[temperature_level],
    "heart_rate_level" : hr_map[heart_rate_level],
    "duration" : dur_map[duration],

    "asthma" : int(asthma),
    "hypertension" : int(hypertension),
    "heart_disease" : int(heart_disease),

    "chief_complaint" : cc_map[chief_complaint]
  }])

patient_scaled = patient.copy()

patient_scaled[cols_to_scale] = scaler.transform(
  patient[cols_to_scale]
)

prediction = model.predict(
  patient_scaled[features]
)[0]

probability = model.predict_proba(
  patient_scaled[features]
)[0]

department = dept_map_inv[prediction]

confidence = probability[prediction] * 100

# tampilkan hasil
st.divider()
st.header("Prediction Result")
info = DEPT_INFO.get(department)

if info:

  st.success(
    f"{info['icon']} : recommended Department : {Department}"
  )
  st.write(f"**Confidence** {confidence: }%")
  st.write("Description: ")
  st.write(info["desc"])
  st.write("What should the patient do ?")

  for step in info["next"]:
    st.write(f"{step}")
else:
  st.success(f"Recomended Department: {department}")
  st.write(f"Confidence : {confidence}")

st.warning("This AI Recomendation is only for educational purposes")










