from firedrake.petsc import PETSc
from preconditioning import build_problem_point_source
from argparse import ArgumentParser
import numpy as np


# parsing command-line arguments
parser = ArgumentParser(description="""Find the amount of GMRES iterations""")
parser.add_argument("--mesh_refinement", type=str,
                    help="Refinement levels of the mesh", required=True)
parser.add_argument("--k", type=str,
                    help="Frequencies k", required=True)
parser.add_argument("--epsilon", type=str,
                    help="Shift preconditioning parameters epsilon", required=True)
parser.add_argument("--file_name", type=str,
                    help="name of csv storing iteration counts", required=True)
args = parser.parse_args()
mesh_refinement_list = [int(i) for i in args.mesh_refinement.split(',')]
k_list = [int(i) for i in args.k.split(',')]
epsilon_list = [int(i) for i in args.epsilon.split(',')]


# iteration counts as a function of k
iteration_array = np.zeros((len(k_list), len(epsilon_list)))

for i, (mesh_refinement, k) in enumerate(zip(mesh_refinement_list, k_list)):
    for j, epsilon in enumerate(epsilon_list):
        PETSc.Sys.Print(f"Point source: amount of GMRES iterations for (mesh_refinement: {mesh_refinement}, k: {k}, epsilon: {epsilon}):")

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

        solver, w = build_problem_point_source(mesh_refinement, parameters, k, epsilon)
        solver.solve()
        iteration_array[i, j] = solver.snes.ksp.getIterationNumber()
        PETSc.Sys.Print(f"{solver.snes.ksp.getIterationNumber()} iterations")

np.savetxt(f"point_source_iterations/{args.file_name}.csv", iteration_array, delimiter=",")