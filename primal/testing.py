import firedrake as fd
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
import argparse
from problems import build_problem_box_source, build_problem_uniform_source, build_problem_sin2
import numpy as np

PETSc.Sys.popErrorHandler()


# parsing command-line arguments
parser = argparse.ArgumentParser(
   description="""Find the amount of GMRES iterations""",
   formatter_class=argparse.ArgumentDefaultsHelpFormatter,
   add_help=False)

parser.add_argument("--help", action="help", default=argparse.SUPPRESS, help=argparse._('show this help message and exit'))
parser.add_argument("--problem", type=str, choices=("box_source", "uniform_source", "sin2"),
                    help="Problem type", required=True)
parser.add_argument("--nx", type=str, choices=("2k", "k^(3/2)"),
                    help="Amount of cells along an edge as a function of k", default="k^(3/2)")
parser.add_argument("--k", type=float, help="Frequency k", required=True)
parser.add_argument("--delta", type=float, help="Shift preconditioning parameter delta", required=True)
parser.add_argument("--delta_0", type=float, help="Shift problem parameter delta_0", default=0)
parser.add_argument("--degree", type=int, help="Degree of CGk", default=2)
parser.add_argument("--sweeps", type=int, help="Maximum amount of multigrid sweeps", default=15)
parser.add_argument("--max_it", type=int, help="Maximum amount of GMRES iterations", default=40)
parser.add_argument("--HSS_method", type=str, choices=("mg", "gamg", "lu"),
                    help="Solver method for HSS iteration", default="mg")
parser.add_argument("--HSS_it", type=str, choices=("k^(1/2)", "k", "k^(3/2)"),
                    help="Amount of HSS iterations as a function of k", default="k")
parser.add_argument("--levels", type=int, help="amount of geometric mg levels", default=2)
parser.add_argument("--m", type=float, help="Multiple for number of HSS iterations", default=1)
parser.add_argument('--HSS_monitor', type=str, choices=("none", "all", "converged_rate", "monitor"),
                    help="Show residuals and converged reason of every HSS iteration")
parser.add_argument('--plot', action="store_true", help="Save plot")
parser.add_argument('--show_args', action="store_true", help="Output all the arguments")
args = parser.parse_known_args()[0]

if args.show_args:  # print args
    PETSc.Sys.Print(args)

k = args.k
delta = args.delta
delta_0 = args.delta_0
sweeps = args.sweeps
max_it = args.max_it
degree = args.degree
levels = args.levels
if args.nx == "2k":  # amount of cells along an edge as a function of k
    nx = int(np.ceil(2*k))
else:
    nx = int(np.ceil(k**(3/2)))

m = args.m
if args.HSS_it == "k^(1/2)":  # amount of HSS iterations as a function of k
    HSS_it = int(np.ceil(m*k**(1/2)))
if args.HSS_it == "k":
    HSS_it = int(np.ceil(m*k))
if args.HSS_it == "k^(3/2)":
    HSS_it = int(np.ceil(m*k**(3/2)))

HSS_monitor = {}
for option in ("monitor", "converged_rate"):
    if args.HSS_monitor in (option, "all"):
        HSS_monitor[option] = None

# defining solver parameters
amg_parameters = {
    "ksp_type": "richardson",
    "ksp_max_it": 5,
    "pc_type": "bjacobi",
    "pc_sub_type": "ilu"
}

star_parameters = {
    'ksp_type': 'chebyshev',
    'pc_type': 'python',
    'pc_python_type': 'firedrake.AssembledPC',
    'assembled': {
        'pc_type': 'python',
        'pc_python_type': 'firedrake.ASMStarPC',
        'pc_star_construct_dim': 0,
        'pc_star_sub_sub_pc_type': 'lu',
    }
}

jacobi_parameters = {
    "ksp_converged_skip": None,
    "ksp_max_it": 3,
    "ksp_type": "chebyshev",
    "pc_type": "jacobi",
}

mg_parameters = jacobi_parameters

helmhss_parameters = {
    "ksp_type": "gmres",
    # "ksp_rtol": 1e-100,
    "ksp_atol": 1e-100,
    "ksp_stol": 1e-100,
    # "ksp_min_it": sweeps,
    "ksp_max_it": sweeps,
    "ksp_converged_skip": None,
    "ksp_converged_maxits": None,
    "pc_type": args.HSS_method,
    "mat_type": "matfree" if args.HSS_method == "mg" else "aij",
    "pc_mg_type": "multiplicative",
    "pc_mg_cycle_type": "w",
    "mg_transfer_manager": "helmholtz.LinearTransferManager",
    "mg_levels": mg_parameters if args.HSS_method == "mg" else amg_parameters,
    "mg_coarse": {
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled_pc_type": "lu",
        "assembled_pc_factor_mat_solver_type": "mumps",
        "assembled_pc_factor_mat_ordering_type": "rcm",
    },
    "its": HSS_it,
    "ksp": HSS_monitor
}

parameters = {
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "ksp": {
        "monitor": None,
        "converged_rate": None,
        "rtol": 1e-5,
        "max_it": max_it,
        "converged_maxits": None,
    },
    "pc_type": "python",
    "pc_python_type": "helmholtz.HSS_PC",
    "helmhss": helmhss_parameters,
}

# creating the linear variational solver
if args.problem == "box_source":
    solver, w = build_problem_box_source(nx, parameters, k, delta, delta_0, degree, args.HSS_method, levels)
if args.problem == "uniform_source":
    solver, w = build_problem_uniform_source(nx, parameters, k, delta, delta_0, degree, args.HSS_method, levels)
if args.problem == "sin2":
    solver, w = build_problem_sin2(nx, parameters, k, delta, delta_0, degree, args.HSS_method, levels)

ndofs = w.function_space().dim()
nranks = w.comm.size
PETSc.Sys.Print(f"Number of processors: {nranks}")
PETSc.Sys.Print(f"Degrees of freedom: {ndofs}")
PETSc.Sys.Print(f"Degrees of freedom per core: {ndofs/nranks}\n")
PETSc.Sys.Print(f"Total floating point numbers: {2*ndofs}")
PETSc.Sys.Print(f"Total floating point numbers per core: {2*ndofs/nranks}\n")

# solving
PETSc.Sys.Print("Solving...")
stime = MPI.Wtime()
solver.solve()
etime = MPI.Wtime()
duration = etime - stime
PETSc.Sys.Print(f"Solver time: {duration}")

if args.plot: # save plot
    file = fd.File(f"plots/{args.problem}_{nx}_{int(k)}_{int(delta)}/plot.pvd")
    file.write(w)
