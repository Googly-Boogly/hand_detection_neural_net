import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import numpy as np


# Define your sequences (19 sequences of 3 integers each)
sequences = [
    [1, 2, 3],
    [4, 5, 6],
    # Add your remaining sequences here
]

# Convert sequences to a numpy array

sequences = np.array(sequences)

# Define the model
model = Sequential()

# Add a SimpleRNN layer with 32 units (you can adjust this as needed)
model.add(SimpleRNN(32, input_shape=(3, 1)))  # The input_shape should match your sequence length and feature count

# Add a Dense layer with 3 output neurons
model.add(Dense(3))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Summary of the model architecture
model.summary()

# Train the model with your data
# You need to have your training data and labels (corresponding to the 3 output neurons)
# Adjust epochs and batch_size as needed
model.fit(sequences, labels, epochs=50, batch_size=1)
