from model import NeuralNetwork
from utils import dfx, prep_data, test_model
from integrator import Integrator
import numpy as np
import torch


def fun(x, LL):  # Boundary condition
    Vb = 2*pow(np.cosh(x*LL), -2)
    return Vb


def dif_loss(model, X_c, T_c, max):  # partial derivatives and equation
    XX = X_c.clone().detach().requires_grad_(True)
    TT = T_c.clone().detach().requires_grad_(True)
    u = model(XX, TT)
    du_dx = dfx(XX, u)
    du_dxx = dfx(XX, du_dx)
    du_dxxx = dfx(XX, du_dxx)
    du_dt = dfx(TT, u)
    return L*L*L/T*du_dt + 6*max*L*L*u*du_dx + du_dxxx


def loss_fun(model, u_b_pred, Vb, X_collocation, T_collocation, max,
             integrator):
    f = dif_loss(model, X_collocation, T_collocation, max)
    error_diffEq = integrator.crude_mc(f)
    error_b = integrator.crude_mc(u_b_pred-Vb)
    return error_b + 1e-6*error_diffEq  # 1e-7 regularisation term


if __name__ == "__main__":
    Nb, Nc = 1200, 800
    L, T = 5, 0.5  # space-time distance
    model = NeuralNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    epochs = 10000
    X_b, T_b, V_b, X_coll, T_coll, max = prep_data(fun, Nb, Nc, L, T)
    integr = Integrator(0, L, Nc)
    for epoch in range(epochs):
        U_b_pred = model(X_b, T_b)
        loss = loss_fun(model, U_b_pred, V_b, X_coll, T_coll, max, integr)
        if epoch % 100 == 99:
            print('epoch', epoch + 1, ", loss: ", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_model(model, 1, L, T=0.5)
