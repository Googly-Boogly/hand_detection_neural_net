import torch
import torch.nn as nn
import numpy as np

# Define your sequences (19 sequences of 19 time steps, each with 42 integers)
sequences = [
    [[1, 2, 3, ..., 42],  # Replace ... with the remaining integers for each time step
     [4, 5, 6, ..., 42],
     # Add your remaining time steps here
    ],
    # Add your remaining sequences here
]

# Convert sequences to a numpy array
sequences = np.array(sequences)

# Convert sequences to a PyTorch tensor
sequences = torch.tensor(sequences, dtype=torch.float)

# Define the model
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init()
        self.gru = nn.GRU(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        output = self.fc(gru_out[-1, :, :])  # Take the output from the last time step
        return output

# Define the model
model = MyModel(input_size=42, hidden_size=32, output_size=3)

# Print model summary
print(model)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model with your data
# You need to have your training data and labels (corresponding to the 3 output neurons)
# Adjust epochs and batch_size as needed
labels = torch.randn((19, 3))  # Replace with your actual labels
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(sequences)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch + 1}/50], Loss: {loss.item():.4f}')
