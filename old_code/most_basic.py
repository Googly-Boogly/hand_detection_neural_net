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


# def test_neural():
#
#     data = data_file_to_code('total_data.txt')
#     labels = data_file_to_code('total_labels.txt')
#     data_dict = prune_data(data, labels)
#     end_data = []
#     end_labels = []
#     # print(len(data_dict['data']))
#     for single_data in data_dict['data']:
#         try:
#             # test_data_structure(single_data)
#             temp_data = training_data_to_neural_network_ready_data(single_data)
#             testing_output = testing_output_from_training_data_to_neural_network_ready_data(temp_data)
#             # print(testing_output)
#             if isinstance(testing_output, list):
#                 end_data.append(testing_output)
#                 test_neural_network_data(testing_output)
#                 print(model(testing_output))
#         except Exception as e:
#             pass
#     temp_labels = labels_to_final_label_ready_for_neural_network(labels)
#     labels = torch.tensor(temp_labels, dtype=torch.float)
#     return end_data
#     # testing_neural_net(end_data, labels)