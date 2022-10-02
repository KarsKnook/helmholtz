from firedrake.petsc import PETSc
from preconditioning import build_problem_5_2
from argparse import ArgumentParser
import numpy as np


# parsing command-line arguments
parser = ArgumentParser(description="""Find the amount of GMRES iterations""")
parser.add_argument("--mesh_refinement", type=str, choices=("2k", "k^3/2"),
                    help="Refinement level of the mesh", required=True)
parser.add_argument("--k", type=str,
                    help="Frequencies k", required=True)
parser.add_argument("--delta", type=str,
                    help="Shift preconditioning parameters delta", required=True)
parser.add_argument("--file_name", type=str,
                    help="name of csv storing iteration counts", required=True)
args = parser.parse_args()
k_list = [float(i) for i in args.k.split(',')]
delta_list = [float(i) for i in args.delta.split(',')]

if args.mesh_refinement == "2k":
    mesh_refinement_list = [int(2*k) for k in k_list]
else: 
    mesh_refinement_list = [int(k**(3/2))+1 for k in k_list]


# iteration counts as a function of k
iteration_array = np.zeros((len(k_list), len(delta_list)))

for i, (mesh_refinement, k) in enumerate(zip(mesh_refinement_list, k_list)):
    for j, delta in enumerate(delta_list):
        PETSc.Sys.Print(f"Problem 5.2: amount of GMRES iterations on "
                    +f"{mesh_refinement}x{mesh_refinement} UnitSquareMesh for k={k} and delta={delta}")

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
            "helmhss_fieldsplit_1_ksp_type": "richardson",
            "helmhss_fieldsplit_1_ksp_convergence_test": "skip",
            "helmhss_fieldsplit_1_ksp_max_it": 15,
            "helmhss_fieldsplit_1_pc_type": "python",
            "helmhss_fieldsplit_1_pc_python_type": "preconditioning.Schur",
            "helmhss_fieldsplit_1_aux_pc_type": "gamg",
            "helmhss_fieldsplit_1_aux_pc_mg_type": "multiplicative",
            "helmhss_fieldsplit_1_aux_pc_mg_cycle_type": "w",
            "helmhss_fieldsplit_1_aux_mg_levels": amg_parameters,
            "helmhss_mat_type": "nest",
            "helmhss_its": int(np.ceil(k)),
            "mat_type": "matfree",
            #"ksp_monitor": None,
            #"ksp_view": None,
        }

        solver, w = build_problem_5_2(mesh_refinement, parameters, k, delta)
        solver.solve()
        iteration_array[i, j] = solver.snes.ksp.getIterationNumber()
        PETSc.Sys.Print(f"{solver.snes.ksp.getIterationNumber()} iterations")

np.savetxt(f"problem_5_2_iterations/{args.file_name}.csv", iteration_array, delimiter=",")