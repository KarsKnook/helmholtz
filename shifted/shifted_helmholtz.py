import firedrake as fd
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from math import ceil, pow
import numpy as np
np.random.seed(12345)

Print = PETSc.Sys.Print

import argparse

parser = argparse.ArgumentParser(
   description="""Solve the Helmholtz problem for a single HSS iteration.""",
   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--mesh_factor", type=int, help="Coefficient c for mesh size c*k^(3/2).", default=1)
parser.add_argument("--k", type=float, help="Frequency k", required=True)
parser.add_argument("--delta", type=float, help="Shift parameter delta", required=True)
parser.add_argument("--degree", type=int, help="Degree of CGk", default=2)
parser.add_argument("--sweeps", type=int, help="Maximum amount of multigrid sweeps", default=15)
parser.add_argument("--mg_it", type=int, help="Maximum amount of multigrid iterations", default=40)
parser.add_argument("--smooth_it", type=int, help="Number of smoother applications on each multigrid level.", default=5)
parser.add_argument("--HSS_method", type=str, choices=("gamg", "lu"), help="Solver method for HSS iteration", default="gamg")
parser.add_argument('--plot', action="store_true", help="Save plot")
parser.add_argument('--output_dir', type=str, default="", help="Directory to write output into.")
parser.add_argument('--show_args', action="store_true", help="Output all the arguments")
args = parser.parse_known_args()[0]

if args.show_args:  # print args
    PETSc.Sys.Print(args)

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


Print("")
Print("### === --- Setting up --- === ###")
Print("")

stime = MPI.Wtime()


nx = args.mesh_factor*int(ceil(pow(args.k, 3/2)))

mesh = fd.UnitSquareMesh(nx, nx)

V = fd.FunctionSpace(mesh, "CG", degree=args.degree)

f = fd.Function(V)
# f.assign(uniform_source(mesh))
for dat in f.dat:
    dat.data[:] = np.random.rand(*(dat.data.shape))

ndofs = V.dim()
ranks = fd.COMM_WORLD.size
Print(f"k = {args.k}, delta = {args.delta}")
Print("")

Print(f"Number of processors: {ranks}")
Print(f"Degrees of freedom: {ndofs}")
Print(f"Total floating point numbers: {2*ndofs}")
Print(f"DoFs/processor: {ndofs/ranks}")
Print(f"FPs/processor: {2*ndofs/ranks}")
Print("")

u = fd.TrialFunction(V)
v = fd.TestFunction(V)

k = fd.Constant(args.k)
delta = fd.Constant(args.delta)

L = fd.inner(f, v)*fd.dx

# A = helmholtz_lhs(u, v, k, delta)
A = hss_lhs(u, v, k, delta)

lu_params = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    # 'pc_factor_mat_ordering_type': 'rcm',
}

amg_parameters = {
    "ksp_type": "gmres",
    "ksp_max_it": args.smooth_it,
    "pc_type": "bjacobi",
    "pc_sub_type": "ilu"
}

mg_params = {
    "ksp_type": "fgmres",
    "ksp_atol": 1e-100,
    "ksp_stol": 1e-100,
    "ksp_max_it": args.mg_it,
    # "ksp_min_it": 15,
    # "ksp_converged_skip": None,
    # "ksp_converged_maxits": None,
    "pc_type": 'gamg',
    "pc_mg_type": "full",
    "pc_mg_cycle_type": "v",
    "mg_levels": amg_parameters,
    "mg_coarse": lu_params,
}

params = {
    'ksp_rtol': 1e-5,
    'ksp_monitor': None,
    'ksp_converged_rate': None,
}

if args.HSS_method == 'lu':
   params.update(lu_params)
elif args.HSS_method == 'gamg':
   params.update(mg_params)

etime = MPI.Wtime()
Print(f"Setup time: {etime-stime} seconds")
Print("")

stime = MPI.Wtime()
w = fd.Function(V, name="w").assign(0)
problem = fd.LinearVariationalProblem(A, L, w)
solver = fd.LinearVariationalSolver(problem, solver_parameters=params)
etime = MPI.Wtime()
Print(f"LVP/LVS creation time: {etime-stime} seconds")
Print("")

Print("### === --- Starting solve --- === ###")
Print("")

stime = MPI.Wtime()
solver.solve()
etime = MPI.Wtime()

Print("")
Print(f"Solution time: {etime-stime} seconds")
Print("")

if args.plot:
    fd.File(f"{args.output_dir}/shifted-k{args.k}-d{args.delta}.pvd").write(w)
