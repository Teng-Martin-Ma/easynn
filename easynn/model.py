import torch
from torch import nn


class Model(nn.Module):
    """Dense neural network model.

    Args:
        input_size (int): input size of the model
        hidden_sizes (list): list of hidden layer sizes

    Returns:
        model (torch.nn.Module): dense neural network model
    """

    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        self.input = nn.Linear(input_size, hidden_sizes[0])
        self.hiddens = nn.ModuleList([
            nn.Linear(hidden_sizes[i], hidden_sizes[i+1])
            for i in range(len(hidden_sizes) - 1)])
        self.output = nn.Linear(hidden_sizes[-1], 1)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.input(x)
        x = self.act(x)
        for layer in self.hiddens:
            x = layer(x)
            x = self.act(x)
        return self.output(x)


@torch.no_grad()
def init_params(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
