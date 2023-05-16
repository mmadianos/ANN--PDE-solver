import torch
import torch.nn as nn


class NeuralNetwork(torch.nn.Module):
    def __init__(
            self, hidden_layers=[20, 20], dim_in=2,
            dim_out=1, activation=torch.tanh):
        '''Initialiser method for Neural Network

        Args:
        '''
        super().__init__()  # Call from superclass

        self.activation = activation
        layers = [dim_in] + hidden_layers + [dim_out]
        self.num_layers = len(layers)
        self.layers = nn.ModuleList([
            nn.Linear(layers[i], layers[i+1])
            for i in range(self.num_layers-1)])

    def forward(self, x, t):  # forward pass on NN with activation function
        # tanh acivation function
        u = torch.cat([x, t], dim=1)

        for i in range(self.num_layers-1):
            u = self.activation(self.layers[i](u))
        return u
