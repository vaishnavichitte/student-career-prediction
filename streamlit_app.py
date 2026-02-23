import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv(r"C:\Users\KINDHA APPLE\student_career_dataset.csv")

X = df.drop("career", axis=1)
y = df["career"]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# UI Design
st.title("🎓 Student Career Prediction App")

st.header("Enter Student Details")

math = st.slider("Math Marks", 0, 100)
science = st.slider("Science Marks", 0, 100)
biology = st.slider("Biology Marks", 0, 100)
english = st.slider("English Marks", 0, 100)
social = st.slider("Social Marks", 0, 100)
computer = st.slider("Computer Marks", 0, 100)

interest_coding = st.slider("Interest in Coding", 0, 10)
interest_biology = st.slider("Interest in Biology", 0, 10)
interest_business = st.slider("Interest in Business", 0, 10)
interest_creativity = st.slider("Interest in Creativity", 0, 10)
interest_public_service = st.slider("Interest in Public Service", 0, 10)

logical_skill = st.slider("Logical Skill", 0, 10)
communication_skill = st.slider("Communication Skill", 0, 10)
leadership_skill = st.slider("Leadership Skill", 0, 10)

if st.button("Predict Career"):

    values = [
        math, science, biology, english, social, computer,
        interest_coding, interest_biology, interest_business,
        interest_creativity, interest_public_service,
        logical_skill, communication_skill, leadership_skill
    ]

    input_df = pd.DataFrame([values], columns=X.columns)

    prediction = model.predict(input_df)
    career_name = le.inverse_transform(prediction)

    prob = model.predict_proba(input_df)
    confidence = np.max(prob) * 100

    st.success(f"Suggested Career: {career_name[0]}")
    st.info(f"Confidence Level: {confidence:.2f}%") 