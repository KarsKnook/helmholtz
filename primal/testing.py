import firedrake as fd
from firedrake.petsc import PETSc
from argparse import ArgumentParser, BooleanOptionalAction
from problems import build_problem_box_source, build_problem_constant, build_problem_sin2
import matplotlib.pyplot as plt


# parsing command-line arguments
parser = ArgumentParser(description="""Find the amount of GMRES iterations""")
parser.add_argument("--problem", type=str, choices=("box_source", "constant", "sin2"),
                    help="Problem type", required=True)
parser.add_argument("--mesh_refinement", type=str, choices=("2k", "k^(3/2)"),
                    help="Refinement level of the mesh as a function of k", required=True)
parser.add_argument("--k", type=float, help="Frequency k", required=True)
parser.add_argument("--delta", type=float, help="Shift preconditioning parameter delta", required=True)
parser.add_argument("--delta_0", type=float, help="Shift problem parameter delta_0", default=0)
parser.add_argument("--degree", type=int, help="Degree of CGk", default=2)
parser.add_argument('--plot', action=BooleanOptionalAction, help="Save plot")
parser.add_argument('--show_args', action=BooleanOptionalAction, help="Output all the arguments")
args = parser.parse_args()

k = args.k
delta = args.delta
delta_0 = args.delta_0
degree = args.degree

if args.mesh_refinement == "2k":  # mesh refinement as a function of k
    mesh_refinement = int(2*k)
else: 
    mesh_refinement = int(k**(3/2)) + 1

if args.show_args:
    PETSc.Sys.Print(args)

# creating the linear variational solver
amg_parameters = {
    "ksp_type": "richardson",
    "ksp_max_it": 5,
    "pc_type": "bjacobi",
    "pc_sub_type": "ilu"
}

helmhss_parameters = {
    "ksp_type": "richardson",
    "ksp_max_it": 15,
    "pc_type": "gamg",
    "pc_mg_type": "multiplicative",
    "pc_mg_cycle_type": "w",
    "mg_levels": amg_parameters,
}

parameters = {
    "ksp_type": "gmres",
    "ksp_rtol": 1e-6,
    "pc_type": "python",
    "pc_python_type": "helmholtz.HSS_PC",
    "helmhss": helmhss_parameters,
    "helmhss_mat_type": "nest",
    "helmhss_its": int(k),
    "mat_type": "matfree",
    "ksp_monitor": None,
}

if args.problem == "box_source":
    solver, w = build_problem_box_source(mesh_refinement, parameters, k, delta, delta_0, degree)
if args.problem == "constant":
    solver, w = build_problem_constant(mesh_refinement, parameters, k, delta, delta_0, degree)
if args.problem == "sin2":
    solver, w = build_problem_sin2(mesh_refinement, parameters, k, delta, delta_0, degree)

# solving
solver.solve()
PETSc.Sys.Print(f"Solver converged in {solver.snes.ksp.getIterationNumber()} GMRES iterations")

# save plot
if args.plot:
    file = fd.File(f"plots/{args.problem}_{mesh_refinement}_{int(k)}_{int(delta)}/plot.pvd")
    file.write(w)