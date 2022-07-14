from firedrake import *
import numpy as np
from helmholtz import solution_plot, error_grid_plot


# parameter values
N = 10
degree = 2
k = 1
epsilon = 1
iters = 10

#initialising
mesh = UnitSquareMesh(N, N)
V = VectorFunctionSpace(mesh, "DG", degree=degree-1, dim=2)
Q = FunctionSpace(mesh, "CG", degree=degree)

x, y = SpatialCoordinate(mesh)
u_exact = sin(pi*x)**2*sin(pi*y)**2

def f_function(mesh, epsilon, k):
    x, y = SpatialCoordinate(mesh)
    return ((-epsilon + k*1j)**2*sin(pi*x)**2*sin(pi*y)**2
            + pi**2*(cos(2*pi*(x+y)) + cos(2*pi*(x-y)) - cos(2*pi*x) - cos(2*pi*y)))

#plotting
#solution_plot(mesh, V, Q, k, epsilon, iters, u_exact, f_function)

epsilon_arr = np.logspace(-2,3,21)
k_arr = np.logspace(-2,3,21)
error_grid_plot(mesh, V, Q, k_arr, epsilon_arr, iters, u_exact, f_function)
