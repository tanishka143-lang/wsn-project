
   

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.title("📡 WSN Sensor Data Prediction")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    
    # Read dataset
    data = pd.read_csv(uploaded_file)
    
    st.write("📊 Dataset Preview")
    st.write(data.head())

    # Extract columns
    X = data["Time"].values.reshape(-1, 1)
    y = data["Temperature"].values

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    # Error calculation
    mse = mean_squared_error(y, predictions)
    st.write(f"📉 Mean Squared Error: {mse:.4f}")

    # Plot graph
    fig, ax = plt.subplots()
    ax.plot(X, y, label="Actual Data")
    ax.plot(X, predictions, label="Predicted Data")
    ax.legend()

    st.pyplot(fig)

else:
    st.info("👆 Please upload a CSV file with Time and Temperature columns")
