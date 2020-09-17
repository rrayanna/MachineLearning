# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Get the Data for Facebook Stock Price Predictions
training_data = pd.read_csv('/TimeSeries/FB_training_data.csv')
test_data = pd.read_csv('/TimeSeries/FB_test_data.csv')

# Convert to a Numpy Array
training_data = training_data.iloc[:, 1].values

# Standardize the training data
scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data.reshape(-1, 1))

# Take 40 days history to predict the future values
x_training_data = []
y_training_data =[]
for i in range(40, len(training_data)):
    x_training_data.append(training_data[i-40:i, 0])
    y_training_data.append(training_data[i, 0])

x_training_data = np.array(x_training_data)
y_training_data = np.array(y_training_data)
x_training_data = np.reshape(x_training_data, (x_training_data.shape[0], x_training_data.shape[1], 1))


## Ready for Tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

# Build the Model
rnn = Sequential()
rnn.add(LSTM(units = 45, return_sequences = True, input_shape = (x_training_data.shape[1], 1)))
rnn.add(Dropout(0.2))
for i in [True, True, False]:
    rnn.add(LSTM(units = 45, return_sequences = i))
    rnn.add(Dropout(0.2))
rnn.add(Dense(units = 1))

# Compile the Model
rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Train the model
rnn.fit(x_training_data, y_training_data, epochs = 100, batch_size = 32)

# Making Predictions
test_data = test_data.iloc[:, 1].values
print(test_data.shape)
plt.plot(test_data)

unscaled_training_data = pd.read_csv('/TimeSeries/FB_training_data.csv')
unscaled_test_data = pd.read_csv('/TimeSeries/FB_test_data.csv')
all_data = pd.concat((unscaled_training_data['Open'], unscaled_test_data['Open']), axis = 0)
x_test_data = all_data[len(all_data) - len(test_data) - 40:].values
x_test_data = np.reshape(x_test_data, (-1, 1))

x_test_data = scaler.transform(x_test_data)
print(x_test_data.shape)
final_x_test_data = []
for i in range(40, len(x_test_data)):
    final_x_test_data.append(x_test_data[i-40:i, 0])

final_x_test_data = np.array(final_x_test_data)
#Reshaping the NumPy array to meet TensorFlow standards
final_x_test_data = np.reshape(final_x_test_data, (final_x_test_data.shape[0], final_x_test_data.shape[1], 1))

predictions = rnn.predict(final_x_test_data)
unscaled_predictions = scaler.inverse_transform(predictions)
plt.clf() #This clears the first prediction plot from our canvas
plt.plot(unscaled_predictions)

plt.plot(unscaled_predictions, color = '#135485', label = "Predictions")
plt.plot(test_data, color = 'black', label = "Real Data")
plt.title('Facebook Stock Price Predictions')