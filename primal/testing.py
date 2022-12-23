import firedrake as fd
from firedrake.petsc import PETSc
from argparse import ArgumentParser, BooleanOptionalAction
from problems import build_problem_box_source, build_problem_constant, build_problem_sin2
import numpy as np


# parsing command-line arguments
parser = ArgumentParser(description="""Find the amount of GMRES iterations""")
parser.add_argument("--problem", type=str, choices=("box_source", "constant", "sin2"),
                    help="Problem type", required=True)
parser.add_argument("--mesh_refinement", type=str, choices=("2k", "k^(3/2)"),
                    help="Mesh refinement as a function of k", required=True)
parser.add_argument("--k", type=float, help="Frequency k", required=True)
parser.add_argument("--delta", type=float, help="Shift preconditioning parameter delta", required=True)
parser.add_argument("--delta_0", type=float, help="Shift problem parameter delta_0", default=0)
parser.add_argument("--degree", type=int, help="Degree of CGk", default=2)
parser.add_argument("--sweeps", type=int, help="Maximum amount of multigrid sweeps", default=15)
parser.add_argument("--max_it", type=int, help="Maximum amount of GMRES iterations", default=100)
parser.add_argument("--HSS_method", type=str, choices=("gamg", "lu"),
                    help="Solver method for HSS iteration", default="gamg")
parser.add_argument("--HSS_it", type=str, choices=("k^(1/2)", "k", "k^(3/2)"),
                    help="Amount of HSS iterations as a function of k", default="k")
parser.add_argument('--plot', action=BooleanOptionalAction, help="Save plot")
parser.add_argument('--show_args', action=BooleanOptionalAction, help="Output all the arguments")
args = parser.parse_args()

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

if args.show_args:  # print args
    PETSc.Sys.Print(args)


# defining solver parameters
amg_parameters = {
    "ksp_type": "richardson",
    "ksp_max_it": 5,
    "pc_type": "bjacobi",
    "pc_sub_type": "ilu"
}

helmhss_parameters = {
    "ksp_type": "richardson",
    "ksp_max_it": sweeps,
    "pc_type": args.HSS_method,
    "pc_mg_type": "multiplicative",
    "pc_mg_cycle_type": "w",
    "mg_levels": amg_parameters,
    "mat_type": "nest",
    "its": HSS_it,
}

parameters = {
    "ksp_type": "gmres",
    "ksp_max_it": max_it,
    "ksp_rtol": 1e-6,
    "pc_type": "python",
    "pc_python_type": "helmholtz.HSS_PC",
    "helmhss": helmhss_parameters,
    "mat_type": "matfree",
    "ksp_monitor": None,
    "ksp_converged_reason": None
}


# creating the linear variational solver
if args.problem == "box_source":
    solver, w = build_problem_box_source(mesh_refinement, parameters, k, delta, delta_0, degree)
if args.problem == "constant":
    solver, w = build_problem_constant(mesh_refinement, parameters, k, delta, delta_0, degree)
if args.problem == "sin2":
    solver, w = build_problem_sin2(mesh_refinement, parameters, k, delta, delta_0, degree)


# solving
solver.solve()

if args.plot: # save plot
    file = fd.File(f"plots/{args.problem}_{mesh_refinement}_{int(k)}_{int(delta)}/plot.pvd")
    file.write(w)