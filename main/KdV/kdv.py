import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import grad
cuda = torch.device('cuda')


Nb = 1200
Nc = 800

L = 5  #space distance
T = 0.5 #time distance

def fun(x, LL):   #Boundary condition
  Vb = 2*pow(np.cosh(x*LL), -2)
  return Vb

def diff_equation(u, du_dt, du_dx, du_dxx, du_dxxx, max):  #Equation-->Regularisation term
  eq = L*L*L/T*du_dt + 6*max*L*L*u*du_dx + du_dxxx
  return eq

def prep_data(Nb, Nc): #Creation of Nb boundary points on [-1,1]x[0,0]/Creation of Nc collocation points on [-1,1]x[0,1]
  r = np.linspace(-1, 1, Nb)
  r1 = np.reshape(r, (Nb, 1))

  r = np.zeros(Nb)
  r2 = np.reshape(r, (Nb, 1))

  V_b = fun(r1, L)
  max = np.amax(V_b)
  V_b = V_b/max
  a = 2*np.random.rand(Nc, 1)-1
  b = np.random.rand(Nc, 1)

  X_coll = torch.from_numpy(a).float()
  T_coll = torch.from_numpy(b).float()

  X_b = torch.from_numpy(r1).float()
  T_b = torch.from_numpy(r2).float()
  V_b = torch.from_numpy(V_b).float()
#print(torch.cat([X_b, T_b, V_b], dim=1))
  return X_b, T_b, V_b, X_coll, T_coll, max

class NeuralNetwork(torch.nn.Module):
    def __init__(self):  # Superclass
        super().__init__()  # Call from superclass

        self.fc1 = torch.nn.Linear(2, 20) #2 inputs: x,t (spacetime)
        self.fc2 = torch.nn.Linear(20, 20)
        self.fc3 = torch.nn.Linear(20, 20)
        self.fc4 = torch.nn.Linear(20, 20)
        self.fc5 = torch.nn.Linear(20, 20)
        self.fc6 = torch.nn.Linear(20, 20)
        self.fc7 = torch.nn.Linear(20, 20)
        self.fc8 = torch.nn.Linear(20, 1) #1 output->equation solution

    def forward(self, xx, tt):  # forward pass on NN with activation function
        tt = tt
        xx = xx
        u = torch.tanh(self.fc1(torch.cat([xx, tt], dim=1))) #tanh acivation function
        u = torch.tanh(self.fc2(u))
        u = torch.tanh(self.fc3(u))
        u = torch.tanh(self.fc4(u))
        u = torch.tanh(self.fc5(u))
        u = torch.tanh(self.fc6(u))
        u = torch.tanh(self.fc7(u))
        u = torch.tanh(self.fc8(u))
        return u


model = NeuralNetwork()


def boundary_loss(model, X_b, T_b):  # Predicted solution for Boundary points
    U_b_pred = model(X_b, T_b)
    return U_b_pred

def dif_loss(model, X_c, T_c, max):  # partial derivatives and equation
    XX = X_c.clone().detach().requires_grad_(True)
    TT = T_c.clone().detach().requires_grad_(True)
    u = model(XX, TT)
    du_dx = grad(u, XX, torch.ones_like(XX), create_graph=True, retain_graph=True)[0]
    du_dxx = grad(du_dx, XX, torch.ones_like(XX), create_graph=True, retain_graph=True)[0]
    du_dxxx = grad(du_dxx, XX, torch.ones_like(XX), create_graph=True, retain_graph=True)[0]

    du_dt = grad(u, TT, torch.ones_like(TT), create_graph=True, retain_graph=True)[0]
    dif_equation = diff_equation(u, du_dt, du_dx, du_dxx, du_dxxx, max)
    return dif_equation


def loss_fun(model, u_b_pred, Vb, X_collocation, T_collocation, max):
    f = dif_loss(model, X_collocation, T_collocation, max)
    error_b = torch.nn.functional.mse_loss(u_b_pred, Vb, reduction ='mean')
    error_diffEq = torch.nn.functional.mse_loss(f, 0 * f, reduction ='mean')
    return error_b + 1e-7*error_diffEq  #1e-7 regularisation term
  

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

epochs = 10000
X_b, T_b, V_b, X_coll, T_coll, max = prep_data(Nb, Nc)

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


#Creation of testing sample XX, TT for time TT=0.5 sec

XX = torch.from_numpy(np.reshape(np.linspace(-1, 1, 101), (101, 1))).float()
TT = torch.from_numpy(0.5*np.ones((101, 1))).float()

u_PINN = model(XX, TT)
#print(torch.cat([XX, u_PINN], dim=1))
u_PINN = u_PINN.detach().numpy()

plt.plot(XX*L, 2*u_PINN)
plt.plot(XX*L, 2*pow(np.cosh(XX*L-2*T), -2)) #analytical solution
plt.show()
