import torch.nn as nn
import torch
from Hyper_Parameters import device
from Hyper_Parameters import initialization_reciprocal_re


class CustomCos(nn.Module):
    def forward(self, x):
        return torch.cos(x)


class Train(nn.Module):
    def __init__(self, neuron_num, stretch_layer_num):
        super(Train, self).__init__()
        sf = torch.arange(0, neuron_num, dtype=torch.float32).view(-1, 1)
        sf = 1 + sf * (stretch_layer_num - 1) / (neuron_num - 1)
        self.scale_factors = sf.to(device)

        self.register_parameter("Reciprocal_Re", nn.Parameter(torch.tensor(initialization_reciprocal_re, requires_grad=True, device=device)))

        self.fc1 = nn.Linear(4, neuron_num)
        nn.init.normal_(self.fc1.weight, mean=0, std=1)
        nn.init.zeros_(self.fc1.bias)

        self.fc2 = nn.Sequential(
            CustomCos(),
            nn.Linear(neuron_num, neuron_num),
            nn.Tanh(),
            nn.Linear(neuron_num, neuron_num),
            nn.Tanh(),
            nn.Linear(neuron_num, neuron_num),
            nn.Tanh(),
            nn.Linear(neuron_num, 4)
        )
        for layer in self.fc2:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def calculate_equation_loss(self, x, y, z, t, prediction_result):
        u, v, w, t = prediction_result[:, 0:1], prediction_result[:, 1:2], prediction_result[:, 2:3], prediction_result[:, 3:4]

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]

        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(v_z), create_graph=True)[0]
