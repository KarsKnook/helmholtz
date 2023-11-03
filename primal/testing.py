import firedrake as fd
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
import argparse
from problems import build_problem_box_source, build_problem_uniform_source, build_problem_sin2
import numpy as np


# parsing command-line arguments
parser = argparse.ArgumentParser(
   description="""Find the amount of GMRES iterations""",
   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--problem", type=str, choices=("box_source", "uniform_source", "sin2"),
                    help="Problem type", required=True)
parser.add_argument("--mesh_refinement", type=str, choices=("2k", "k^(3/2)"),
                    help="Mesh refinement as a function of k", required=True)
parser.add_argument("--k", type=float, help="Frequency k", required=True)
parser.add_argument("--delta", type=float, help="Shift preconditioning parameter delta", required=True)
parser.add_argument("--delta_0", type=float, help="Shift problem parameter delta_0", default=0)
parser.add_argument("--degree", type=int, help="Degree of CGk", default=2)
parser.add_argument("--sweeps", type=int, help="Maximum amount of multigrid sweeps", default=15)
parser.add_argument("--max_it", type=int, help="Maximum amount of GMRES iterations", default=40)
parser.add_argument("--HSS_method", type=str, choices=("gamg", "lu"),
                    help="Solver method for HSS iteration", default="gamg")
parser.add_argument("--HSS_it", type=str, choices=("k^(1/2)", "k", "k^(3/2)"),
                    help="Amount of HSS iterations as a function of k", default="k")
parser.add_argument('--HSS_monitor', type=str, choices=("none", "all", "converged_reason", "monitor"),
                    help="Show residuals and converged reason of every HSS iteration")
parser.add_argument('--plot', action="store_true", help="Save plot")
parser.add_argument('--show_args', action="store_true", help="Output all the arguments")
args = parser.parse_known_args()[0]

k = args.k
delta = args.delta
delta_0 = args.delta_0
sweeps = args.sweeps
max_it = args.max_it
degree = args.degree

if args.mesh_refinement == "2k":  # mesh refinement as a function of k
    mesh_refinement = int(np.ceil(2*k))
else:
    mesh_refinement = int(np.ceil(k**(3/2)))

if args.HSS_it == "k^(1/2)":  # amount of HSS iterations as a function of k
    HSS_it = int(np.ceil(k**(1/2)))
if args.HSS_it == "k":
    HSS_it = int(np.ceil(k))
if args.HSS_it == "k^(3/2)":
    HSS_it = int(np.ceil(k**(3/2)))

HSS_monitor = {}
for option in ("monitor", "converged_reason"):
    if args.HSS_monitor in (option, "all"):
        HSS_monitor[option] = None

if args.show_args:  # print args
    PETSc.Sys.Print(args)

# defining solver parameters
amg_parameters = {
    "ksp_type": "richardson",
    "ksp_max_it": 5,
    "pc_type": "bjacobi",
    "pc_sub_type": "ilu"
    # "pc_type": "pbjacobi"
}

helmhss_parameters = {
    "ksp_type": "gmres",
    "ksp_rtol": 1e-100,
    "ksp_atol": 1e-100,
    "ksp_stol": 1e-100,
    "ksp_max_it": sweeps,
    # "ksp_min_it": sweeps,
    "ksp_converged_skip": None,
    "ksp_converged_maxits": None,
    "pc_type": args.HSS_method,
    "pc_mg_type": "multiplicative",
    "pc_mg_cycle_type": "w",
    "mg_levels": amg_parameters,
    # "pc_gamg": {
    #    "repartition": False,
    #    "process_eq_limit": 200,
    #    "coarse_eq_limit": 200,
    # },
    "mat_type": "nest",
    "its": HSS_it,
    "ksp": HSS_monitor
}

parameters = {
    "ksp_type": "fgmres",
    "ksp_max_it": max_it,
    "ksp_rtol": 1e-6,
    "pc_type": "python",
    "pc_python_type": "helmholtz.HSS_PC",
    "helmhss": helmhss_parameters,
    "mat_type": "matfree",
    "ksp_monitor": None,
    "ksp_converged_reason": None,
    #"ksp_view": None,
}

# creating the linear variational solver
if args.problem == "box_source":
    solver, w = build_problem_box_source(mesh_refinement, parameters, k, delta, delta_0, degree)
if args.problem == "uniform_source":
    solver, w = build_problem_uniform_source(mesh_refinement, parameters, k, delta, delta_0, degree)
if args.problem == "sin2":
    solver, w = build_problem_sin2(mesh_refinement, parameters, k, delta, delta_0, degree)

with solver.inserted_options():
   solver.snes.setUp()
   solver.snes.getKSP().setUp()
   solver.snes.getKSP().getPC().setUp()

PETSc.Sys.Print(f"Number of processors: {w.comm.size}")
PETSc.Sys.Print(f"Degrees of freedom: {w.function_space().dim()}")
PETSc.Sys.Print(f"Degrees of freedom per core: {w.function_space().dim()//w.comm.size}\n")

# solving
stime = MPI.Wtime()
solver.solve()
etime = MPI.Wtime()
duration = etime - stime
PETSc.Sys.Print(f"Solver time: {duration}")

if args.plot: # save plot
    file = fd.File(f"plots/{args.problem}_{mesh_refinement}_{int(k)}_{int(delta)}/plot.pvd")
    file.write(w)
