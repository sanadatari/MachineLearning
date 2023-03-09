import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load the dataset
data = pd.read_csv("C:\\Users\\USER\\prices.txt")
data = pd.read_csv("C:\\Users\\USER\\prices.txt", sep ="\t")


# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# Preprocess the data
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# Create the model
model = Sequential()
model.add(LSTM(50, input_shape=(train_scaled.shape[1], 1)))
model.add(Dense(1))

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Train the model
history = model.fit(train_scaled, train_data, epochs=50, batch_size=32, verbose=2)

# Evaluate the model
mse, mae = model.evaluate(test_scaled, test_data, verbose=0)
rmse = np.sqrt(mse)
print('MAE: %.3f, RMSE: %.3f' % (mae, rmse))

# Predict the most recent 20% prices
recent_prices = data.tail(int(0.2 * len(data)))
recent_scaled = scaler.transform(recent_prices)
recent_X = np.reshape(recent_scaled, (recent_scaled.shape[0], recent_scaled.shape[1], 1))
predictions = model.predict(recent_X)
