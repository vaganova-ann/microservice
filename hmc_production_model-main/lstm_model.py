import torch
import torch.nn as nn


def init_hidden_state(batch_size: int, num_layers: int, hidden_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    h0 = torch.zeros(size=(num_layers, batch_size, hidden_size))
    c0 = torch.zeros(size=(num_layers, batch_size, hidden_size))
    return h0, c0


class LstmModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, linear_layer_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Sequential(nn.Linear(hidden_size, linear_layer_size), nn.LeakyReLU(0.05))
        self.output_layer1 = nn.Sequential(nn.Linear(linear_layer_size, output_size), nn.LeakyReLU(0.05))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0, c0 = init_hidden_state(batch_size=x.size(0), num_layers=self.num_layers, hidden_size=self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        # Decode the hidden state of the last time step
        out = self.output_layer(out[:, -1, :])
        out = self.output_layer1(out)
        return out
