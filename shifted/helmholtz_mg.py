import firedrake as fd
from firedrake.petsc import PETSc
from math import sqrt
import numpy as np
np.random.seed(12345)

PETSc.Sys.popErrorHandler()
Print = PETSc.Sys.Print


class RichardsonComplexPC:
    @PETSc.Log.EventDecorator()
    def create(self, ksp):
        self.work = []

    @PETSc.Log.EventDecorator()
    def destroy(self, ksp):
        for vec in self.work:
            if vec:
                vec.destroy()
        self.work = []

    def view(self, ksp, viewer):
        viewer.printfASCII(f'Richardson KSP with complex damping\n')
        viewer.printfASCII(f'  damping factor={self.scale}\n')

    @PETSc.Log.EventDecorator()
    def setUp(self, ksp):
        self.work[:] = ksp.getWorkVecs(right=2, left=None)

    def setFromOptions(self, ksp):
        self.scale = PETSc.Options().getScalar(
            ksp.getOptionsPrefix()+'ksp_richardson_scale', 1)

    @PETSc.Log.EventDecorator()
    def solve(self, ksp, b, x):
        A, _ = ksp.getOperators()
        P = ksp.getPC()
        r, z = self.work
        ksp.setIterationNumber(0)

        def iterate(A, P, b, x, r, z):
            A.mult(x, r)
            r.aypx(-1, b)
            P.apply(r, z)

        if ksp.getInitialGuessNonzero():
            A.mult(x, r)
            r.aypx(-1, b)
        else:
            b.copy(r)
        P.apply(r, z)
        while not self._check_convergence(ksp, z):
            x.axpy(self.scale, z)
            iterate(A, P, b, x, r, z)

    def _check_convergence(self, ksp, r):
        its = ksp.getIterationNumber()
        rnorm = r.norm()
        ksp.setResidualNorm(rnorm)
        ksp.logConvergenceHistory(rnorm)
        ksp.monitor(its, rnorm)
        reason = ksp.callConvergenceTest(its, rnorm)

        if ksp.its >= ksp.max_it:
            reason = PETSc.KSP.ConvergedReason.CONVERGED_ITS
            ksp.setConvergedReason(reason)
        elif reason:
            ksp.setConvergedReason(reason)
        else:
            ksp.setIterationNumber(its+1)

        return reason


class LinearTransferManager(fd.TransferManager):
    def inject(self, source, target):
        return


def omega_opt(k, h, epsilon, gamma):
    # num = 12 - 4*h*h*(k*k + 1j*epsilon)
    # den = 18 - 3*h*h*(k*k + 1j*epsilon)
    # return num/den
    num = -gamma.real*k*k + gamma.imag*epsilon
    den = 1 + abs(k*k + 1j*epsilon)
    return (num/den)**2

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(
   description="""Solve the 1D shifted Helmholtz problem using Jacobi-MG.""",
   formatter_class=ArgumentDefaultsHelpFormatter
)

parser.add_argument("--nxbase", type=int, default=32, help="Number of nodes on the base mesh.")
parser.add_argument("--nref", type=int, default=1, help="Number of mesh refinements.")
parser.add_argument("--k", type=float, default=64, help="Wavenumber.")
parser.add_argument("--C", type=float, default=2, help="Shift coefficient.")
parser.add_argument("--nu", type=int, default=2, help="Number of smoothing steps.")
parser.add_argument("--omega", type=float, help="Damping factor. If not given, CG17 method is used.")
parser.add_argument("--level_omega", action="store_true", help="Level dependent damping factor.")
parser.add_argument('--show_args', action="store_true", help="Output all the arguments")

args, _ = parser.parse_known_args()

if args.show_args:  # print args
    PETSc.Sys.Print(args)

base_mesh = fd.UnitIntervalMesh(args.nxbase)
mesh = fd.MeshHierarchy(base_mesh, args.nref)[-1]

hs = tuple(1/(args.nxbase*pow(2, i)) for i in range(args.nref+1))

nx = args.nxbase*pow(2, args.nref)
h = 1/nx
h_base = 1/args.nxbase

V = fd.FunctionSpace(mesh, 'CG', 1)

u = fd.TrialFunction(V)
v = fd.TestFunction(V)

C = args.C
k = args.k

epsilon = C*k*k
gamma = 0.5 + 1j*sqrt(3)/2

omega = omega_opt(k, h, epsilon, gamma)
omega_base = omega_opt(k, h_base, epsilon, gamma)

omegas = tuple(omega_opt(k, hi, epsilon, gamma) for hi in hs)

# Print(f"{k = } | {nx = } | kh = {round(k*h, 4)} | kh/sqrt(6) = {round(k*h/sqrt(6), 4)} | kh_base/sqrt(6) = {round(k*h_base/sqrt(6), 4)}")
# Print(f"{omega = }")
# Print(f"omega = {omega} | {omega_base = }")

kc = fd.Constant(k)
ec = fd.Constant(epsilon)

shift = kc*kc + 1j*ec

a = shift*fd.inner(u, v)*fd.dx + fd.inner(fd.grad(u), fd.grad(v))*fd.dx

lu_parameters = {
    'ksp_type': 'preonly',
    'pc_type': 'python',
    'pc_python_type': 'firedrake.AssembledPC',
    'assembled_pc_type': 'lu'
}

def complex_str(z, i='i'):
    return f'{z.real}{"+" if z.imag > 0 else ""}{z.imag}{i}'

smooth_parameters = {
    'ksp_norm_type': 'preconditioned',  # stop python_ksp complaining
    'ksp_converged_skip': None,
    'ksp_converged_maxits': None,
    'ksp_type': 'python',
    'ksp_python_type': f'{__name__}.RichardsonComplexPC',
    'ksp_max_it': args.nu,
    'pc_type': 'jacobi',
    # 'pc_jacobi_abs': None,
}

parameters = {
    # 'ksp_monitor': ':ksp_monitor.log',
    # 'ksp_monitor': None,
    'ksp_converged_rate': None,
    'ksp_view': ':ksp_view.log',
    'ksp_rtol': 1e-6,
    'mat_type': 'matfree',
    'ksp_type': 'python',
    'ksp_python_type': f'{__name__}.RichardsonComplexPC',
    'pc_type': 'mg',
    'pc_mg_type': 'multiplicative',
    'pc_mg_cycle_type': 'v',
    'pc_mg_distinct_smoothup': None,
    'mg_transfer_manager': f'{__name__}.LinearTransferManager',
    'mg_levels': smooth_parameters,
    'mg_coarse': smooth_parameters,
    'mg_levels_up': { # only use down-smoother
        'ksp_max_it': 0,
        'ksp_type': 'richardson',
        'pc_type': 'none',
    },
}

if args.omega:
    parameters['mg_coarse_ksp_richardson_scale'] = args.omega
    parameters['mg_levels_ksp_richardson_scale'] = args.omega
elif args.level_omega:
    parameters['mg_coarse_ksp_richardson_scale'] = complex_str(omegas[0])
    for i in range(1,args.nref+1):
        parameters[f'mg_levels_{i}_ksp_richardson_scale'] = complex_str(omegas[i])
else:
    parameters['mg_coarse_ksp_richardson_scale'] = complex_str(omegas[-1])
    parameters['mg_levels_ksp_richardson_scale'] = complex_str(omegas[-1])



L = fd.Cofunction(V.dual())
u = fd.Function(V)
problem = fd.LinearVariationalProblem(a, L, u, constant_jacobian=True)
solver = fd.LinearVariationalSolver(problem, options_prefix='',
                                    solver_parameters=parameters)
for i in range(1):
    L.dat.data[:] = np.random.random_sample(L.dat.data.shape)
    u.dat.data[:] = np.random.random_sample(u.dat.data.shape)
    solver.solve()
