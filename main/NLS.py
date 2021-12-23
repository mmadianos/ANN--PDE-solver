import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import grad
cuda = torch.device('cuda')

Nb = 800
Nc = 1000

L = 5 #space distance
T = 1 #time distance

def funR(x, LL):    #real value of boundary condition
  Vb = pow(np.cosh(x*LL), -1)*np.cos(x*LL) 
  return Vb

def funIm(x, LL):     #imaginary value of boundary condition
  Vb = pow(np.cosh(x*LL), -1)*np.sin(x*LL)
  return Vb

def diff_equationR(u, v, dv_dt, du_dxx, L, T, minR, maxR, minIm, maxIm):   #real value of the diff. equation
  dIm = maxIm - minIm
  dR = maxR - minR
  eqR = -L*L*dv_dt/T+0.5*du_dxx+L*L*(torch.square_(u) + torch.square_(v)) *u
  return eqR

def diff_equationIm(u, v, du_dt, dv_dxx, L, T, minR, maxR, minIm, maxIm):       #imaginary value of the diff. equation
  dIm = maxIm - minIm
  dR = maxR - minR
  eqIm = L*L*du_dt/T+0.5*dv_dxx+L*L*(torch.square_(u) + torch.square_(v)) *v
  return eqIm

def prep_data(Nb, Nc):
  r = np.linspace(-1, 1, Nb)
  r1 = np.reshape(r, (Nb, 1))

  r = np.zeros(Nb)
  r2 = np.reshape(r, (Nb, 1))

  V_b = funR(r1, L)
  maxR = np.amax(V_b)
  minR = np.amin(V_b)
  #V_b = (V_b-minR)/(maxR-minR)

  U_b= funIm(r1, L)
  maxIm = np.amax(U_b)
  minIm = np.amin(U_b)
  #U_b = (U_b-minIm)/(maxIm-minIm)
  V_b = np.hstack((V_b, U_b))

  a = 2*np.random.rand(Nc, 1)-1
  b = np.random.rand(Nc, 1)

  X_coll = torch.from_numpy(a).float()
  T_coll = torch.from_numpy(b).float()

  X_b = torch.from_numpy(r1).float()
  T_b = torch.from_numpy(r2).float()
  V_b = torch.from_numpy(V_b).float()

#print(torch.cat([X_b, T_b, V_b], dim=1))
  return X_b, T_b, V_b, X_coll, T_coll, minR, maxR, minIm, maxIm

class NeuralNetwork(torch.nn.Module):
    def __init__(self):  # Superclass
        super().__init__()  # Call from superclass

        self.fc1 = torch.nn.Linear(2, 20)
        self.fc2 = torch.nn.Linear(20, 40)
        self.fc3 = torch.nn.Linear(40, 40)
        self.fc4 = torch.nn.Linear(40, 20)
        self.fc5 = torch.nn.Linear(20, 20)
        self.fc6 = torch.nn.Linear(20, 20)
        self.fc7 = torch.nn.Linear(20, 20)
        self.fc8 = torch.nn.Linear(20, 2)     #2 outputs: Real and Imaginary part of the solution

    def forward(self, xx, tt):  # forward pass on NN with activation function
        tt = tt
        xx = xx
        u = torch.tanh(self.fc1(torch.cat([xx, tt], dim=1)))
        u = torch.tanh(self.fc2(u))
        u = torch.tanh(self.fc3(u))
        u = torch.tanh(self.fc4(u))
        u = torch.tanh(self.fc5(u))
        u = torch.tanh(self.fc6(u))
        u = torch.tanh(self.fc7(u))
        u = torch.tanh(self.fc8(u))
        return u


model = NeuralNetwork()


def boundary_loss(model, X_b, T_b):  # b for Boundary points
    U_b_pred = model(X_b, T_b)
    return U_b_pred


def dif_loss(model, X_c, T_c, minR, maxR, minIm, maxIm):  # partial derivatives
    XX = X_c.clone().detach().requires_grad_(True)
    TT = T_c.clone().detach().requires_grad_(True)
    u = model(XX, TT)
    uu = torch.reshape(u[:, 0], (Nc, 1)).clone().detach().requires_grad_(False)
    vv = torch.reshape(u[:, 1], (Nc, 1)).clone().detach().requires_grad_(False)

    du_dx = grad(torch.reshape(u[:, 0], (Nc, 1)), XX, torch.ones_like(XX), create_graph=True, retain_graph=True)[0]
    du_dxx = grad(du_dx, XX, torch.ones_like(XX), create_graph=True, retain_graph=True)[0]
    dv_dx = grad(torch.reshape(u[:, 1], (Nc, 1)), XX, torch.ones_like(XX), create_graph=True, retain_graph=True)[0]
    dv_dxx = grad(dv_dx, XX, torch.ones_like(XX), create_graph=True, retain_graph=True)[0]

    du_dt = grad(torch.reshape(u[:, 0], (Nc, 1)), TT, torch.ones_like(TT), create_graph=True, retain_graph=True)[0]
    dv_dt = grad(torch.reshape(u[:, 1], (Nc, 1)), TT, torch.ones_like(TT), create_graph=True, retain_graph=True)[0]

    dif_equation_real = diff_equationR(uu, vv, dv_dt, du_dxx, L, T, minR, maxR, minIm, maxIm)
    dif_equation_im = diff_equationIm(uu, vv, du_dt, dv_dxx, L, T, minR, maxR, minIm, maxIm)
    return torch.cat([dif_equation_real, dif_equation_im], dim=1)


def loss_funR(model, u_b_pred, Vb, minR, maxR, minIm, maxIm, f):
    error_b_real = torch.nn.functional.mse_loss(u_b_pred[:, 0], Vb[:, 0])
    error_diffEq_real = torch.nn.functional.mse_loss(f[:, 0], 0 * f[:, 0])
    #print(error_diffEq_im)
    #print(error_diffEq_real)
    return error_b_real + 5e-5*error_diffEq_real #regularisation term


def loss_funIm(model, u_b_pred, Vb, minR, maxR, minIm, maxIm, f):
    error_b_im = torch.nn.functional.mse_loss(u_b_pred[:, 1], Vb[:, 1])
    error_diffEq_im = torch.nn.functional.mse_loss(f[:, 1], 0 * f[:, 1])
    #print(error_diffEq_im)
    #print(error_diffEq_real)
    return error_b_im + 5e-5*error_diffEq_im 

optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)  # 6

epochs = 15000
X_b, T_b, V_b, X_coll, T_coll, minR, maxR, minIm, maxIm = prep_data(Nb, Nc)

for epoch in range(epochs):
    f = dif_loss(model, X_coll, T_coll, minR, maxR, minIm, maxIm)
    U_b_pred = boundary_loss(model, X_b, T_b)
    lossR = loss_funR(model, U_b_pred, V_b, minR, maxR, minIm, maxIm, f)
    lossIm = loss_funIm(model, U_b_pred, V_b, minR, maxR, minIm, maxIm, f)
    loss = lossR + lossIm
    if epoch % 100 == 99:
        print('epoch', epoch + 1, ", lossR: ", lossR.item())
        print('epoch', epoch + 1, ", lossIm: ", lossIm.item())
    if epoch % 5000 == 4999:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 5
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#Creation of test sample XX,TT: test solution for TT=1 sec
XX = torch.from_numpy(np.reshape(np.linspace(-1, 1, 101), (101, 1))).float()
TT = torch.from_numpy(np.ones((101, 1))).float()

u_PINN = model(XX, TT)

Sol = torch.sqrt_(torch.square_(u_PINN[:, 0])+torch.square_(u_PINN[:, 1]))
Sol = Sol.detach().numpy()

plt.plot(XX*L, Sol)
plt.plot(XX*L, pow(np.cosh(XX*L-T), -1))
plt.show()
