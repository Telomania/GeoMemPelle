import torch
import torch.nn as nn

from reservoir import Reservoir


# Reproducibility
torch.manual_seed(42)

# Model sizes
INPUT_SIZE = 1
HIDDEN_SIZE = 50

# Training settings
NUM_SAMPLES = 1000
EPOCHS = 30
LEARNING_RATE = 0.01


# Generate artificial input sequences
x = torch.randn(NUM_SAMPLES, 5, INPUT_SIZE)

# Target: was the first number positive?
y = (x[:, 0, 0] > 0).float().unsqueeze(1)


# Create the reservoir
reservoir = Reservoir(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
)


# Get reservoir states without training the reservoir
with torch.no_grad():
    _, hidden = reservoir(x)

    # Shape: (number of samples, hidden size)
    features = hidden.squeeze(0)


# Create the trainable readout
readout = nn.Linear(HIDDEN_SIZE, 1)

loss_function = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    readout.parameters(),
    lr=LEARNING_RATE,
)


# Train only the readout
for epoch in range(EPOCHS):

    predictions = readout(features)

    loss = loss_function(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            accuracy = (
                (predictions > 0) == (y > 0.5)
            ).float().mean()

        print(
            f"Epoch {epoch + 1:2d} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy.item():.3f}"
        )