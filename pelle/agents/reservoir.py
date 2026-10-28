import torch
import torch.nn as nn


class Reservoir(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        connectivity=0.1,
        spectral_radius=0.9,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input weights: input -> reservoir
        self.W_in = nn.Parameter(
            torch.randn(hidden_size, input_size),
            requires_grad=False,
        )

        # Create random recurrent connections
        W_res = torch.randn(hidden_size, hidden_size)

        # Keep only a fraction of the connections
        mask = torch.rand(hidden_size, hidden_size) < connectivity
        W_res *= mask

        # Scale the recurrent weights by their spectral radius
        eigenvalues = torch.linalg.eigvals(W_res)
        radius = eigenvalues.abs().max().real

        W_res = W_res * (spectral_radius / radius)

        self.W_res = nn.Parameter(
            W_res,
            requires_grad=False,
        )

    def forward(self, x, state=None):

        # x shape: (batch, sequence_length, input_size)
        batch_size = x.shape[0]

        if state is None:
            state = torch.zeros(
                batch_size,
                self.hidden_size,
                device=x.device,
                dtype=x.dtype,
            )

        outputs = []

        # Process the sequence one time step at a time
        for t in range(x.shape[1]):

            current_input = x[:, t, :]

            state = torch.tanh(
                current_input @ self.W_in.T
                + state @ self.W_res.T
            )

            outputs.append(state)

        output = torch.stack(outputs, dim=1)

        return output, state