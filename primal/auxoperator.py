import firedrake as fd
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
import argparse


class LinearTransferManager(fd.TransferManager):
    def inject(self, source, target):
        return


def helmholtz(u, v, mcoeff, bcoeff, lcoeff=1.):
    mcoeff = mcoeff if isinstance(mcoeff, fd.Constant) else fd.Constant(mcoeff)
    bcoeff = bcoeff if isinstance(bcoeff, fd.Constant) else fd.Constant(bcoeff)
    lcoeff = lcoeff if isinstance(lcoeff, fd.Constant) else fd.Constant(lcoeff)
    return (fd.inner(mcoeff*u, v)*fd.dx
            + fd.inner(bcoeff*u, v)*fd.ds
            + fd.inner(lcoeff*fd.grad(u), fd.grad(v))*fd.dx)


def shifted_helmholtz(u, v, delta, k):
    return helmholtz(u, v,
                     mcoeff=(-delta + 1j*k)**2,
                     bcoeff=(delta - 1j*k))


def hss_lhs(u, v, delta, k):
    kv = k.values()[0]  # avoid UFL warnign (bug) casting to complex to float
    fcoeff1 = fd.Constant((kv+1)/(2*kv))
    return fcoeff1*helmholtz(u, v,
                             mcoeff=(-2j*delta*k**2 + delta**2 - k**2),
                             bcoeff=(delta - 1j*k**2))


class ShiftedHelmholtzPC(fd.AuxiliaryOperatorPC):
    _prefix = "helmshift_"
    needs_python_pmat = True

    def form(self, pc, v, u):
        _, P = pc.getOperators()
        bcs = P.getPythonContext().bcs
        appctx = self.get_appctx(pc)
        k, delta = appctx["k"], appctx["delta"]
        a = shifted_helmholtz(u, v, delta, k)
        return (a, bcs)


class HSSHermitianPC(fd.AuxiliaryOperatorPC):
    _prefix = "hss_"
    needs_python_pmat = True

    def form(self, pc, v, u):
        _, P = pc.getOperators()
        bcs = P.getPythonContext().bcs
        appctx = self.get_appctx(pc)
        k = appctx["k"]
        k, delta = appctx["k"], appctx["delta"]
        a = hss_lhs(u, v, delta, k)
        return (a, bcs)


PETSc.Sys.popErrorHandler()
Print = PETSc.Sys.Print

# parsing command-line arguments
parser = argparse.ArgumentParser(
   description="""Solve the Helmholtz equation using HSS preconditioning and Jacobi-MG.""",
   formatter_class=argparse.ArgumentDefaultsHelpFormatter,
   add_help=False)

parser.add_argument("--help", action="help", default=argparse.SUPPRESS, help=argparse._('Show this help message and exit.'))
parser.add_argument("--base_nx", type=int, help="Number of cells along each edge of the base mesh.", required=True)
parser.add_argument("--levels", type=int, default=2, help="Number of geometric mg levels.")
parser.add_argument("--k", type=float, help="Wavenumber.", required=True)
parser.add_argument("--m", type=int, default=1, help="Shifted preconditioner uses m*k HSS iterations.")
parser.add_argument('--direct_shift', action="store_true", help="Use a direct solver for the shifted Helmholtz preconditioner.")
parser.add_argument('--direct_hss', action="store_true", help="Use a direct solver for the HSS preconditioner.")
parser.add_argument('--direct_coarse', action="store_true", help="Use a direct solver for the coarsest multigrid level.")
parser.add_argument("--delta", type=float, default=1, help="Preconditioner shift parameter.")
parser.add_argument("--delta0", type=float, default=0, help="Problem shift parameter.")
parser.add_argument("--degree", type=int, default=2, help="Degree of CGk.")
parser.add_argument('--verbose', action="store_true", help="Output some diagnostics.")
parser.add_argument('--show_args', action="store_true", help="Output all the arguments.")
args = parser.parse_known_args()[0]

if args.show_args:  # print args
    Print(args)

k = args.k
delta = args.delta
delta0 = args.delta0


# defining solver parameters
def fixed_its(its):
    return {
        "converged_skip": None,
        "converged_maxits": None,
        "max_it": its,
    }


lu_params = {
    'ksp_type': 'preonly',
    'pc_type': 'python',
    'pc_python_type': 'firedrake.AssembledPC',
    'assembled_pc_type': 'lu',
    "assembled_pc_factor_mat_solver_type": "mumps",
}

parameters = {
    "ksp": {
        "view": ":ksp_view.log",
        "monitor": None,
        "converged_rate": None,
        "converged_maxits": None,
        "max_it": 10,
        "rtol": 1e-4,
    },
    "mat_type": "matfree",
    "ksp_type": "gmres",
    "pc_type": "python",
    "pc_python_type": f"{__name__}.ShiftedHelmholtzPC",
    "helmshift": lu_params if args.direct_shift else {
        "mat_type": "matfree",
        "pc_type": "ksp",
        "ksp": {
            "ksp_converged_rate": None,
            "ksp": fixed_its(args.m*int(k)),
            "ksp_type": "richardson",
            "pc_type": "python",
            "pc_python_type": f"{__name__}.HSSHermitianPC",
            "hss": lu_params if args.direct_hss else {
                "mat_type": "matfree",
                "pc_type": "ksp",
                "ksp": {
                    "ksp_converged_rate": ":hss_converged_rate.log",
                    "ksp": fixed_its(2),
                    "ksp_type": "richardson",
                    "pc_type": "mg",
                    "pc_mg_type": "multiplicative",
                    "pc_mg_cycle_type": "w",
                    "mg_transfer_manager": f"{__name__}.LinearTransferManager",
                    "mg_levels": {
                        "ksp": fixed_its(2),
                        "ksp_type": "richardson",
                        "ksp_richardson_scale": 2/3,
                        "pc_type": "jacobi",
                    },
                    "mg_coarse": lu_params if args.direct_coarse else {
                        "ksp": fixed_its(2),
                        "ksp_type": "richardson",
                        "ksp_richardson_scale": 2/3,
                        "pc_type": "jacobi",
                    },
                }
            },
        }
    },
}

base_mesh = fd.UnitSquareMesh(args.base_nx, args.base_nx)
mesh = base_mesh if args.levels == 1 else fd.MeshHierarchy(base_mesh, args.levels-1)[-1]

V = fd.FunctionSpace(mesh, "CG", degree=args.degree)

ndofs = V.dim()
nranks = mesh.comm.size
if args.verbose:
    Print(f"Number of processors: {nranks}")
    Print(f"Degrees of freedom: {ndofs}")
    Print(f"Degrees of freedom per core: {ndofs/nranks}")
    Print(f"Total floating point numbers: {2*ndofs}")
    Print(f"Total floating point numbers per core: {2*ndofs/nranks}\n")

# helmholtz form
u = fd.TrialFunction(V)
v = fd.TestFunction(V)

k_c = fd.Constant(k)
delta_c = fd.Constant(delta)
delta0_c = fd.Constant(delta0)

A = shifted_helmholtz(u, v, delta0_c, k_c)

appctx = {"k": k_c, "delta": delta_c}

# uniform source
L = fd.assemble(fd.inner(fd.Constant(1), v)*fd.dx)
w = fd.Function(V)

problem = fd.LinearVariationalProblem(A, L, w)
solver = fd.LinearVariationalSolver(
    problem, appctx=appctx, options_prefix="",
    solver_parameters=parameters)

# solving
stime = MPI.Wtime()
solver.solve()
etime = MPI.Wtime()
duration = etime - stime
if args.verbose:
    Print(f"Solver time: {duration}")
    Print("Expected convergence bounds:")
    Print(f"Shift->Helmholtz convergence: c*delta^0.5 = c*{delta**2}")
    Print(f"HSS->Shift convergence: {(1-1/k)/(1+1/k) = }")
