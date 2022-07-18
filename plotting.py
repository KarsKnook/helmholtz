from firedrake import *
import numpy as np
from helmholtz import solution_plot, error_grid_plot, convergence_plot


# parameter values
N = 10
degree = 2
k = 1
epsilon = 10
iters = 100

#initialising
mesh = UnitSquareMesh(N, N)
V = VectorFunctionSpace(mesh, "DG", degree=degree-1, dim=2)
Q = FunctionSpace(mesh, "CG", degree=degree)

x, y = SpatialCoordinate(mesh)
u_exact = sin(pi*x)**2*sin(pi*y)**2

def f_function(mesh, k, epsilon):
    x, y = SpatialCoordinate(mesh)
    return ((-epsilon + k*1j)**2*sin(pi*x)**2*sin(pi*y)**2
            + pi**2*(cos(2*pi*(x+y)) + cos(2*pi*(x-y)) - cos(2*pi*x) - cos(2*pi*y)))

f = f_function(mesh, k, epsilon)

#plotting
#solution_plot(mesh, V, Q, f, k, epsilon, iters)

"""k_arr = np.logspace(-2,3,6)
epsilon_arr = np.logspace(-2,3,6)
error_grid_plot(mesh, V, Q, f_function, k_arr, epsilon_arr, iters, u_exact)"""

k_arr = np.array([1000])
epsilon_arr = np.array([1])
convergence_plot(mesh, V, Q, f_function, k_arr, epsilon_arr, iters=1000)