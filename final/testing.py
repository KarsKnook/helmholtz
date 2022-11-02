import firedrake as fd
from firedrake.petsc import PETSc
from argparse import ArgumentParser, BooleanOptionalAction
from problems import build_problem_box_source, build_problem_constant, build_problem_sin2


# parsing command-line arguments
parser = ArgumentParser(description="""Find the amount of GMRES iterations""")
parser.add_argument("--problem", type=str, choices=("box_source", "constant", "sin2"),
                    help="Problem type", required=True)
parser.add_argument("--mesh_refinement", type=str, choices=("2k", "k^(3/2)"),
                    help="Refinement level of the mesh", required=True)
parser.add_argument("--k", type=float, nargs=1,
                    help="Frequency k", required=True)
parser.add_argument("--delta", type=float, nargs=1,
                    help="Shift preconditioning parameter delta", required=True)
parser.add_argument('--plot', action=BooleanOptionalAction, help="Save plot")
parser.add_argument('--show_args', action=BooleanOptionalAction, help="Output all the arguments")
args = parser.parse_args()

k = args.k[0]
delta = args.delta[0]

if args.mesh_refinement == "2k":
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

parameters = {
    "ksp_type": "gmres",
    "ksp_rtol": 1e-6,
    "pc_type": "python",
    "pc_python_type": "helmholtz.HSS_PC",
    "helmhss_ksp_type": "preonly",
    "helmhss_pc_type": "fieldsplit",
    "helmhss_pc_fieldsplit_type": "schur",
    "helmhss_pc_fieldsplit_schur_fact_type": "full",
    "helmhss_fieldsplit_0_ksp_type": "preonly",
    "helmhss_fieldsplit_0_pc_type": "bjacobi",
    "helmhss_fieldsplit_0_sub_pc_type": "ilu",
    "helmhss_fieldsplit_1_ksp_type": "richardson",
    "helmhss_fieldsplit_1_ksp_max_it": 15,
    "helmhss_fieldsplit_1_pc_type": "python",
    "helmhss_fieldsplit_1_pc_python_type": "helmholtz.Schur",
    "helmhss_fieldsplit_1_aux_pc_type": "gamg",
    "helmhss_fieldsplit_1_aux_pc_mg_type": "multiplicative",
    "helmhss_fieldsplit_1_aux_pc_mg_cycle_type": "w",
    "helmhss_fieldsplit_1_aux_mg_levels": amg_parameters,
    "helmhss_mat_type": "nest",
    "helmhss_its": int(k),
    "mat_type": "matfree",
    "ksp_monitor": None,
}

if args.problem == "box_source":
    solver, w = build_problem_box_source(mesh_refinement, parameters, k, delta)
if args.problem == "constant":
    solver, w = build_problem_constant(mesh_refinement, parameters, k, delta)
if args.problem == "sin2":
    solver, w = build_problem_sin2(mesh_refinement, parameters, k, delta)

# solving
solver.solve()

# save plot
if args.plot:
    sigma, u = w.split()
    file = fd.File(f"plots/{args.problem}_{mesh_refinement}_{int(k)}_{int(delta)}/plot.pvd")
    file.write(sigma, u)