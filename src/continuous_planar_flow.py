import torch
import torch.nn as nn

from torchdiffeq import odeint_adjoint as odeint
# from torchdiffeq import odeint

import numpy as np

class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, solver='dopri5', atol=1e-5, rtol=1e-5):
        super(CNF, self).__init__()

        self.odefunc = odefunc
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.times = torch.tensor([0., T])

    def forward(self, z_0, logpz_0):

        z_t, logpz_t = odeint(
            self.odefunc,
            (z_0, logpz_0),
            self.times.to(z_0),
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver,
        )

        z_t, logpz_t = z_t[1], logpz_t[1]


        return z_t, logpz_t


def trace_df_dz(f, z):
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()

# def trace_df_dz(f, z):
#     jac = _get_minibatch_jacobian(f, z)
#     diagonal = jac.view(jac.shape[0], -1)[:, ::jac.shape[1]]
#     return torch.sum(diagonal, 1)


# def _get_minibatch_jacobian(y, x):
#     '''
#     Compute the Jacobian matrix in batch form.
#     Return (B, D_y, D_x)
#     '''

#     batch = y.shape[0]
#     single_y_size = np.prod(y.shape[1:])
#     y = y.view(batch, -1)
#     vector = torch.ones(batch).to(y)

#     # Compute Jacobian row by row.
#     # dy_i / dx -> dy / dx
#     # (B, D) -> (B, 1, D) -> (B, D, D)
#     jac = [torch.autograd.grad(y[:, i], x, 
#                                grad_outputs=vector, 
#                                retain_graph=True,
#                                create_graph=True)[0].view(batch, -1)
#                 for i in range(single_y_size)]
#     jac = torch.stack(jac, dim=1)
    
#     return jac


# def _get_minibatch_jacobian(y, x):
#     """Computes the Jacobian of y wrt x assuming minibatch-mode.
#     Args:
#       y: (N, ...) with a total of D_y elements in ...
#       x: (N, ...) with a total of D_x elements in ...
#     Returns:
#       The minibatch Jacobian matrix of shape (N, D_y, D_x)
#     """
#     assert y.shape[0] == x.shape[0]
#     y = y.view(y.shape[0], -1)

#     # Compute Jacobian row by row.
#     jac = []
#     for j in range(y.shape[1]):
#         dy_j_dx = torch.autograd.grad(y[:, j], x, torch.ones_like(y[:, j]), retain_graph=True,
#                                       create_graph=True)[0].view(x.shape[0], -1)
#         jac.append(torch.unsqueeze(dy_j_dx, 1))
#     jac = torch.cat(jac, 1)
#     return jac

# def trace_df_dz(f, z):
#     df_dz = torch.autograd.grad(f, z)
#     print(df_dz.shape)
#     return torch.trace(df_dz)


class ContinuousPlanarFlow(nn.Module):

    def __init__(self, in_out_dim, width):
        super(ContinuousPlanarFlow, self).__init__()

        self.diffeq = nn.Sequential(
            nn.Linear(in_out_dim, width),
            nn.Softplus(),
            nn.Linear(width, in_out_dim, bias=False),
        )

    def forward(self, t, states):
        z = states[0]
        # z = states

        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            dz_dt = self.diffeq(z)
            dlogpz_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, dlogpz_dt)
        # return (dz_dt,)
        # return dz_dt


# class LinearSimple(nn.Module):
#     def __init__(self, dim):
#         super(LinearSimple, self).__init__()
#         self.lin = nn.Linear(dim, dim, bias=False)

#     def forward(self, t, x):
#         return torch.tanh(self.lin(x))


# class WidePlanarFlow(nn.Module):
#     def __init__(self, in_out_dim, hidden_dim):
#         super(WidePlanarFlow, self).__init__()
#         self.lin1 = nn.Linear(in_out_dim, hidden_dim)
#         self.lin2 = nn.Linear(hidden_dim, in_out_dim, bias=False)
#         self.act = nn.Softplus()

#     def forward(self, x):
#         h = self.lin1(x)
#         return self.lin2(self.act(h))
