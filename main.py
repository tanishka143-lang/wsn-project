import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create data
time = np.arange(0, 100, 0.5)
temperature = 25 + np.sin(time) + np.random.normal(0, 0.5, len(time))

data = pd.DataFrame({
    'time': time,
    'temp': temperature
})

# Lag features
def create_lag(data):
    df = data.copy()
    df['lag1'] = df['temp'].shift(1)
    df['lag2'] = df['temp'].shift(2)
    df['lag3'] = df['temp'].shift(3)
    df = df.dropna()
    return df

df = create_lag(data)

# Train model
X = df[['lag1', 'lag2', 'lag3']]
y = df['temp']

model = LinearRegression()
model.fit(X, y)

# Prediction logic
threshold = 0.8
transmissions = 0
saved = 0

predictions = []
actuals = []

for i in range(3, len(data)):
    lag_values = data['temp'][i-3:i].values.reshape(1, -1)
    pred = model.predict(lag_values)[0]
    actual = data['temp'][i]

    if abs(pred - actual) > threshold:
        transmissions += 1
    else:
        saved += 1

    predictions.append(pred)
    actuals.append(actual)

print("Transmissions:", transmissions)
print("Saved:", saved)

# Plot graph
plt.plot(actuals, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.title("Prediction vs Actual")
plt.show()