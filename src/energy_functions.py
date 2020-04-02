"""Test energy functions from "Variational Inference with Normalizing Flows" by Rezende and Mohamed (2015)"""

import math
import torch


def energy_function_1(z):
    return _squared_frac(torch.norm(z, dim=1, keepdim=True) - 2, 0.4) - \
     torch.log(_squared_exp(z[:, 0:1], 2, 0.6) + _squared_exp(z[:, 0:1], -2, 0.6))
    # Norm contant for -4 to 4: 0.10209194971849117


def energy_function_2(z):
    return _squared_frac(z[:, 1:2] - _w_1(z), 0.4)
    # Norm contant for -4 to 4: 0.12530008870938594


def energy_function_3(z):
    return -torch.log(_squared_exp(z[:, 1:2], _w_1(z), 0.35) +
                      _squared_exp(z[:, 1:2] + _w_2(z), _w_1(z), 0.35))
    # Norm contant for -4 to 4: 0.21927515522245916


def energy_function_4(z):
    return -torch.log(_squared_exp(z[:, 1:2], _w_1(z), 0.4) +
                      _squared_exp(z[:, 1:2] + _w_3(z), _w_1(z), 0.35))
    # Norm contant for -4 to 4: 0.22888885693701488


def _w_1(z):
    return torch.sin(2 * math.pi * z[:, 0:1] / 4)


def _w_2(z):
    return 3 * _squared_exp(z[:, 0:1], 1, 0.6)


def _w_3(z):
    return 3 * torch.sigmoid((z[:, 0:1] - 1) / 0.3)


def _squared_exp(a, b, c):
    return torch.exp(-_squared_frac(a - b, c))


def _squared_frac(num, den):
    return 0.5 * ((num / den) ** 2)
