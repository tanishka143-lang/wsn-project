import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.title("📡 WSN Sensor Data Prediction")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:

    # Read file
    data = pd.read_csv(uploaded_file)

    st.write("📊 Raw Data")
    st.write(data.head())

    # 🔥 FIX 1: Check columns
    st.write("Columns in dataset:", data.columns)

    # 🔥 FIX 2: Ensure correct column names
    if "Time" not in data.columns or "Temperature" not in data.columns:
        st.error("❌ CSV must contain 'Time' and 'Temperature' columns")
    
    else:
        # 🔥 FIX 3: Clean data
        data = data[["Time", "Temperature"]]
        data = data.dropna()
      

        # Convert to numeric
        data["Time"] = pd.to_numeric(data["Time"], errors='coerce')
        data["Temperature"] = pd.to_numeric(data["Temperature"], errors='coerce')

        data = data.dropna()
          st.write("Number of rows:", len(data))

        st.write("✅ Cleaned Data")
        st.write(data.head())

        # 🔥 FIX 4: Extract properly
        X = data[["Time"]]   # IMPORTANT (double brackets)
        y = data["Temperature"]

        # 🔥 FIX 5: Train model
        model = LinearRegression()
        model.fit(X, y)

        predictions = model.predict(X)
        threshold = 1.0  # you can adjust this

transmissions = 0
saved = 0

for i in range(len(y)):
    error = abs(y.iloc[i] - predictions[i])
    
    if error > threshold:
        transmissions += 1
    else:
        saved += 1

st.write(f"📡 Transmissions: {transmissions}")
st.write(f"💾 Saved: {saved}")

        # Error
        mse = mean_squared_error(y, predictions)
        st.write(f"📉 Mean Squared Error: {mse:.4f}")

        # Plot
        fig, ax = plt.subplots()
        ax.plot(data["Time"], y, label="Actual")
        ax.plot(data["Time"], predictions, label="Predicted")
        ax.legend()

        st.pyplot(fig)

else:
    st.info("👆 Upload a CSV file with Time and Temperature")
