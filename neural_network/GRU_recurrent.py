import datetime

import torch
import torch.nn as nn
import numpy as np
from neural_network.GRU_model import MyModel
from neural_network.data_transformation import data_file_to_code, labels_to_final_label_ready_for_neural_network, total_data_to_sequences, unit_test_sequences_data, str_to_float_for_nn, standardize_data
from neural_network.data_augmentation import AugmentData
from neural_network.helpful_functions import create_logger_error, log_it
import os


def train_neural_network(sequences_data, labels):
    # Convert sequences to a NumPy array
    sequences = np.array(sequences_data)

    # Convert sequences to a PyTorch tensor
    sequences = torch.tensor(sequences, dtype=torch.float)
    sequences = sequences.view(-1, 1, 43)  # Update the shape to match your sequences

    # Define the model
    model = MyModel(input_size=43, hidden_size=32, output_size=3)

    # Print model summary
    print(model)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model with your data
    for epoch in range(50):
        cur = datetime.datetime.now()
        optimizer.zero_grad()
        outputs = model(sequences)
        for i in range(outputs.shape[0]):
            loss = criterion(outputs[i:i + 1], labels[i:i + 1])
            loss.backward()
        optimizer.step()
        print(f"time for 1 epoch {cur-datetime.datetime.now()}")
        print(f'Epoch [{epoch + 1}/50], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'trained_model.pth')


def run_neural_network():
    data = data_file_to_code('data_files/data.txt')
    labels2 = data_file_to_code('data_files/labels.txt')
    sequences_data2 = total_data_to_sequences(data)
    sequences_data3 = []
    for dat in sequences_data2:
        temp_idk = standardize_data(dat)
        if isinstance(temp_idk, list):
            sequences_data3.append(temp_idk)
    sequences_data = []
    for temp2 in sequences_data3:
        check = False
        for temp3 in temp2:
            if len(temp3) != 43:
                check = True
        if not check:
            sequences_data.append(temp2)
        else:
            check = False
    new_labels = []
    for label in labels2:
        temp_var = str_to_float_for_nn(label)
        if temp_var == 1:
            new_labels.append([1.0, 0, 0])
        if temp_var == 2:
            new_labels.append([0, 1.0, 0])
        if temp_var == 3:
            new_labels.append([0, 0, 1.0])
    for seq in sequences_data:
        unit_test_sequences_data(seq)


    total_data = []
    total_labels = []
    runner = 0
    logger = create_logger_error(os.path.abspath(__file__), 'run_neural_network')
    while runner < len(sequences_data):
        print(len(total_data))
        try:
            individual_label = new_labels[runner]
            individual_data = sequences_data[runner]
            # print(individual_data)
            if len(individual_data) > 2:
                partial_data = AugmentData(
                        data=individual_data,
                        rotation_range=10,
                        scaling_range=0.1,
                        translation_range=0.1,
                        noise_level=0.1,
                        joint_dropout_prob=0.01,
                        )
                partial_data.augment_data()
                for inp in partial_data.new_data:
                    total_data.append(inp)
                    total_labels.append(individual_label)
        except Exception as e:
            log_it(logger, e)
        runner += 1
    for temp4 in total_data:
        unit_test_sequences_data(temp4)
    labels = torch.tensor(total_labels, dtype=torch.float)
    # print(labels)
    train_neural_network(total_data, labels)


def load_model():
    model = MyModel(input_size=42, hidden_size=32, output_size=3)
    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()
    # print(model)
    return model


if __name__ == '__main__':
    pass
    # run_neural_network()
