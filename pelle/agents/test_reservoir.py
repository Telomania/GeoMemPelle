import torch

from reservoir import Reservoir


torch.manual_seed(42)

reservoir = Reservoir(
    input_size=4,
    hidden_size=50,
    connectivity=0.1,
    spectral_radius=0.9,
)

x = torch.randn(1, 5, 4)

output, final_state = reservoir(x)

print("Output shape:", output.shape)
print("Final state shape:", final_state.shape)

print(
    "Recurrent weights require gradients:",
    reservoir.W_res.requires_grad,
)

print(
    "Input weights require gradients:",
    reservoir.W_in.requires_grad,
)

print(
    "Actual spectral radius:",
    torch.linalg.eigvals(reservoir.W_res).abs().max().item(),
)