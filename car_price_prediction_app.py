import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
    df = pd.read_csv(url)
    df.dropna(inplace=True)
    return df

# Data Preprocessing
def preprocess_data(df):
    # Encode all categorical variables automatically
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Select features and target
    X = df.drop(['mpg'], axis=1)
    y = df['mpg']
    return X, y

# Model Training
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def main():
    st.title("Car Price Prediction using Linear Regression")

    # Load and display data
    df = load_data()
    st.write("### Dataset Preview")
    st.write(df.head())

    # Preprocessing
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = train_model(X_train, y_train)

    # Model evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write("### Model Evaluation")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    # User Input for Prediction
    st.write("### Make a Prediction")
    user_data = []
    for col in X.columns:
        val = st.number_input(f"Enter value for {col}")
        user_data.append(val)

    if st.button("Predict"):
        try:
            result = model.predict([user_data])
            st.write(f"Predicted Value: {result[0]:.2f}")
        except ValueError as e:
            st.error("Invalid input! Please make sure all inputs match the required format.")

if __name__ == "__main__":
    main()
