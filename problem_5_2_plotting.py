import firedrake as fd
import matplotlib.pyplot as plt
from preconditioning import build_problem_5_2
from argparse import ArgumentParser


# parsing command-line arguments
parser = ArgumentParser(description="""Find the amount of GMRES iterations""")
parser.add_argument("mesh_refinement", type=int, nargs=1,
                    help="Refinement level of the mesh")
parser.add_argument("k", type=float, nargs=1,
                    help="Frequency k")
parser.add_argument("delta", type=float, nargs=1,
                    help="Shift preconditioning parameters delta")
args = parser.parse_args()
mesh_refinement = args.mesh_refinement[0]
k = args.k[0]
delta = args.delta[0]


#plotting the solution
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
    "pc_python_type": "preconditioning.pHSS_PC",
    "helmhss_ksp_type": "preonly",
    "helmhss_pc_type": "fieldsplit",
    "helmhss_pc_fieldsplit_type": "schur",
    "helmhss_pc_fieldsplit_schur_fact_type": "full",
    "helmhss_fieldsplit_0_ksp_type": "preonly",
    "helmhss_fieldsplit_0_pc_type": "bjacobi",
    "helmhss_fieldsplit_0_sub_pc_type": "ilu",
    "helmhss_fieldsplit_1_ksp_type": "preonly",
    "helmhss_fieldsplit_1_pc_type": "python",
    "helmhss_fieldsplit_1_pc_python_type": "preconditioning.Schur",
    "helmhss_fieldsplit_1_aux_pc_type": "gamg",
    "helmhss_fieldsplit_1_aux_pc_mg_type": "multiplicative",
    "helmhss_fieldsplit_1_aux_pc_mg_cycle_type": "w",
    "helmhss_fieldsplit_1_aux_mg_levels": amg_parameters,
    "helmhss_mat_type": "nest",
    "helmhss_its": int(k),
    "mat_type": "matfree",
    #"ksp_monitor": None,
    #"ksp_view": None,
}

solver, w = build_problem_5_2(mesh_refinement, parameters, k, delta)
solver.solve()

sigma, u = w.split()
file = fd.File("problem_5_2_plots/problem_5_2_plot.pvd")
file.write(sigma, u)