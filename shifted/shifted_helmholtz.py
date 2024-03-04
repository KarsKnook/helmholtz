import firedrake as fd
from firedrake.petsc import PETSc
from math import ceil, pow
import numpy as np
np.random.seed(12345)

Print = PETSc.Sys.Print


def helmholtz_lhs(u, v, k, delta):
    A = ((-delta + 1j*k)**2)*fd.inner(u, v)*fd.dx \
        + fd.inner(fd.grad(u), fd.grad(v))*fd.dx \
        - (-delta + 1j*k)*fd.inner(u, v)*fd.ds
    return A


def hss_lhs(u, v, k, delta):
    a = (-2j*delta*k**2 + delta**2 - k**2)*fd.inner(u, v)*fd.dx \
        + fd.inner(fd.grad(u), fd.grad(v))*fd.dx \
        + (-1j*k**2 + delta)*fd.inner(u, v)*fd.ds
    return a


def uniform_source(mesh):
    return 1


def box_source(mesh):
    x, y = fd.SpatialCoordinate(mesh)
    f = fd.conditional(fd.ge(x, 0.4), 1, 0)
    f *= fd.conditional(fd.le(x, 0.6), 1, 0)
    f *= fd.conditional(fd.ge(y, 0.4), 1, 0)
    f *= fd.conditional(fd.le(y, 0.6), 1, 0)
    return f


K = 64
Delta = 1

nx = int(ceil(pow(K, 3/2)))

mesh = fd.UnitSquareMesh(nx, nx)

V = fd.FunctionSpace(mesh, "CG", degree=2)

f = fd.Function(V)
# f.assign(uniform_source(mesh))
for dat in f.dat:
    dat.data[:] = np.random.rand(*(dat.data.shape))

ndofs = V.dim()
Print(f"Degrees of freedom: {ndofs}")
Print(f"Total floating point numbers: {2*ndofs}")

Print(f"k = {K}, delta = {Delta}")

u = fd.TrialFunction(V)
v = fd.TestFunction(V)

k = fd.Constant(K)
delta = fd.Constant(Delta)

L = fd.inner(f, v)*fd.dx

# A = helmholtz_lhs(u, v, k, delta)
A = hss_lhs(u, v, k, delta)

lu_params = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'pc_factor_mat_ordering_type': 'rcm',
}

amg_parameters = {
    "ksp_type": "gmres",
    "ksp_max_it": 5,
    "pc_type": "bjacobi",
    "pc_sub_type": "ilu"
}

mg_params = {
    "ksp_type": "fgmres",
    "ksp_atol": 1e-100,
    "ksp_stol": 1e-100,
    "ksp_max_it": 15,
    # "ksp_min_it": 15,
    # "ksp_converged_skip": None,
    # "ksp_converged_maxits": None,
    "pc_type": 'gamg',
    "pc_mg_type": "full",
    "pc_mg_cycle_type": "v",
    "mg_levels": amg_parameters,
    "mg_coarse": lu_params,
}

monitor = {
    'ksp_rtol': 1e-5,
    'ksp_monitor': None,
    'ksp_converged_rate': None,
}

params = mg_params
params.update(monitor)

w = fd.Function(V, name="w").assign(0)
problem = fd.LinearVariationalProblem(A, L, w)
solver = fd.LinearVariationalSolver(problem, solver_parameters=params)

solver.solve()
fd.File(f"plots/shifted-k{K}-d{Delta}.pvd").write(w)
