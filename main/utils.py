import torch
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt


def dfx(x, f):
    '''
    Calculate the derivative with auto-differention
    '''
    return grad(
        [f], [x], grad_outputs=torch.ones(x.shape, dtype=torch.float),
        create_graph=True, retain_graph=True)[0]


def prep_data(fun, Nb, Nc, L, T):
    '''
    Creation of Nb boundary points on [-1,1]x[0,0]
    Creation of Nc collocation points on [-1,1]x[0,1]
    '''
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
# print(torch.cat([X_b, T_b, V_b], dim=1))
    return X_b, T_b, V_b, X_coll, T_coll, max


def test_model(model, analytical_solution, L, T):
    # Creation of testing sample XX, TT for time TT=0.5 sec

    XX = torch.from_numpy(np.reshape(
        np.linspace(-1, 1, 101), (101, 1))).float()
    TT = torch.from_numpy(T*np.ones((101, 1))).float()

    u_PINN = model(XX, TT)
    # print(torch.cat([XX, u_PINN], dim=1))
    u_PINN = u_PINN.detach().numpy()

    plt.plot(XX*L, 2*u_PINN)
    plt.plot(XX*L, 2*pow(np.cosh(XX*L-2*T), -2))  # analytical solution
    plt.savefig('image')
