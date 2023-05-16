from model import NeuralNetwork
from utils import dfx, prep_data, test_model
import numpy as np
import torch


Nb, Nc = 1200, 800
L, T = 5, 0.5  # space-time distance


def fun(x, LL):  # Boundary condition
    Vb = 2*pow(np.cosh(x*LL), -2)
    return Vb


def diff_equation(u, du_dt, du_dx, du_dxx, du_dxxx, max):
    # Equation-->Regularisation term
    eq = L*L*L/T*du_dt + 6*max*L*L*u*du_dx + du_dxxx
    return eq


model = NeuralNetwork()


def boundary_loss(model, X_b, T_b):  # Predicted solution for Boundary points
    U_b_pred = model(X_b, T_b)
    return U_b_pred


def dif_loss(model, X_c, T_c, max):  # partial derivatives and equation
    XX = X_c.clone().detach().requires_grad_(True)
    TT = T_c.clone().detach().requires_grad_(True)
    u = model(XX, TT)
    du_dx = dfx(XX, u)
    du_dxx = dfx(XX, du_dx)
    du_dxxx = dfx(XX, du_dxx)
    du_dt = dfx(TT, u)
    return diff_equation(u, du_dt, du_dx, du_dxx, du_dxxx, max)


def loss_fun(model, u_b_pred, Vb, X_collocation, T_collocation, max):
    f = dif_loss(model, X_collocation, T_collocation, max)
    error_b = torch.nn.functional.mse_loss(u_b_pred, Vb, reduction='mean')
    error_diffEq = torch.nn.functional.mse_loss(f, 0 * f, reduction='mean')
    return error_b + 1e-6*error_diffEq  # 1e-7 regularisation term


optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

epochs = 10000
X_b, T_b, V_b, X_coll, T_coll, max = prep_data(fun, Nb, Nc, L, T)

for epoch in range(epochs):
    U_b_pred = boundary_loss(model, X_b, T_b)
    loss = loss_fun(model, U_b_pred, V_b, X_coll, T_coll, max)
    if epoch % 100 == 99:
        print('epoch', epoch + 1, ", loss: ", loss.item())
    if epoch % 5000 == 4999:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


test_model(model, 1, L, T=0.5)
