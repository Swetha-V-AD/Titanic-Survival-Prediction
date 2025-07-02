import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load data and train model (simplified version)
@st.cache_data
def train_model():
    df = pd.read_csv('Titanic-Dataset.csv')  # or train.csv if renamed
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'], inplace=True)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# UI Layout
st.title("🚢 Titanic Survival Prediction")

Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 100, 25)
SibSp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
Parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
Fare = st.number_input("Fare", 0.0, 500.0, 50.0)
Embarked_Q = st.checkbox("Embarked at Queenstown?")
Embarked_S = st.checkbox("Embarked at Southampton?")

# Preprocess input
sex_code = 0 if Sex == "male" else 1
embarked_q = 1 if Embarked_Q else 0
embarked_s = 1 if Embarked_S else 0

input_data = np.array([[Pclass, sex_code, Age, SibSp, Parch, Fare, embarked_q, embarked_s]])

if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("🎯 Passenger would have SURVIVED")
    else:
        st.error("💀 Passenger would NOT have survived")

