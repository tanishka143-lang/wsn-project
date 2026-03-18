import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("WSN Data Prediction System")

if st.button("Run Simulation"):

    time = np.arange(0, 100, 0.5)
    temperature = 25 + np.sin(time) + np.random.normal(0, 0.5, len(time))

    data = pd.DataFrame({
        'time': time,
        'temp': temperature
    })

    def create_lag(data):
        df = data.copy()
        df['lag1'] = df['temp'].shift(1)
        df['lag2'] = df['temp'].shift(2)
        df['lag3'] = df['temp'].shift(3)
        df = df.dropna()
        return df

    df = create_lag(data)

    X = df[['lag1', 'lag2', 'lag3']]
    y = df['temp']

    model = LinearRegression()
    model.fit(X, y)

    threshold = 0.8
    transmissions = 0
    saved = 0

    predictions = []
    actuals = []

    for i in range(3, len(data)):
        lag_values = pd.DataFrame([data['temp'][i-3:i].values], columns=['lag1','lag2','lag3'])
        pred = model.predict(lag_values)[0]
        actual = data['temp'][i]

        if abs(pred - actual) > threshold:
            transmissions += 1
        else:
            saved += 1

        predictions.append(pred)
        actuals.append(actual)

    st.write("Transmissions:", transmissions)
    st.write("Saved:", saved)

    fig, ax = plt.subplots()
    ax.plot(actuals, label="Actual")
    ax.plot(predictions, label="Predicted")
    ax.legend()

    st.pyplot(fig)