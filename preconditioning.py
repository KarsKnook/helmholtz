from firedrake import *
import matplotlib.pyplot as plt


def build_problem(mesh_size, parameters, k, epsilon, aP=None, block_matrix=False):
    mesh = UnitSquareMesh(mesh_size, mesh_size)

    V = VectorFunctionSpace(mesh, "DG", degree=1, dim=2)
    Q = FunctionSpace(mesh, "CG", degree=2)
    W = V * Q

    sigma_old = interpolate(Constant((0,0), mesh), V)
    u_old = interpolate(Constant(0, mesh), Q)

    sigma_new, u_new = TrialFunctions(W)
    tau, v = TestFunctions(W)
    f = Function(V)

    def f_function(mesh, k, epsilon):
        x, y = SpatialCoordinate(mesh)
        return ((-epsilon + k*1j)**2*sin(pi*x)**2*sin(pi*y)**2
                + pi**2*(cos(2*pi*(x+y)) + cos(2*pi*(x-y)) - cos(2*pi*x) - cos(2*pi*y)))

    f = f_function(mesh, k, epsilon)

    a = (inner(Constant((epsilon-1j)*k)*sigma_new, tau)*dx
         - inner(grad(u_new), tau)*dx
         + inner(sigma_new, grad(v))*dx
         + inner(Constant((epsilon-1j)*k)*u_new, v)*dx
         + inner(Constant(k)*u_new,v)*ds)
    L = (Constant((k-1)/(k+1))*(inner(Constant((epsilon+1j)*k)*sigma_old, tau)*dx
                               + inner(grad(u_old), tau)*dx
                               - inner(sigma_old, grad(v))*dx
                               + inner(Constant((epsilon+1j)*k)*u_old, v)*dx
                               + inner(Constant(k)*u_old, v)*ds)
        - Constant(2*k/(k+1))*inner(f/Constant(-epsilon+1j*k), v)*dx)

    if aP is not None:
        aP = aP(W)
    if block_matrix:
        mat_type = 'nest'
    else:
        mat_type = 'aij'

    if aP is not None:
        P = assemble(aP, mat_type=mat_type)
    else:
        P = None

    w = Function(W)
    vpb = LinearVariationalProblem(a, L, w, aP=aP)
    solver =  LinearVariationalSolver(vpb, solver_parameters=parameters)

    return solver, w


class pHSS_PC(preconditioners.base.PCBase):

    needs_python_pmat = True

    def initialize(self, pc):
        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + "pHSSp_"
        # we assume P has things stuffed inside of it
        _, P = pc.getOperators()
        context = P.getPythonContext()

        k = context.appctx.get("k", 1.0)
        epsilon = context.appctx.get("epsilon", 1.0)

        test, trial = context.a.arguments()

        if test.function_space() != trial.function_space():
            raise ValueError("pHSS_PC only makes sense if test and trial space are the same")

        W = test.function_space()

        sigma_new, u_new = TrialFunctions(W)
        tau, v = TestFunction(W)
        # Handle vector and tensor-valued spaces.

        a = Constant((epsilon-1j*k)*(k+1)/(2*k))*(inner(Constant((epsilon-1j)*k)*sigma_new, tau)*dx
                                                  - inner(grad(u_new), tau)*dx
                                                  + inner(sigma_new, grad(v))*dx
                                                  + inner(Constant((epsilon-1j)*k)*u_new, v)*dx
                                                  + inner(Constant(k)*u_new,v)*ds)

        opts = petsc.PETSc.Options()
        mat_type = opts.getString(options_prefix + "mat_type",
                                  parameters["default_matrix_type"])

        A = assemble(a, form_compiler_parameters=context.fc_params,
                     mat_type=mat_type, options_prefix=options_prefix)

        Pmat = A.petscmat
        Pmat.setNullSpace(P.getNullSpace())
        tnullsp = P.getTransposeNullSpace()
        if tnullsp.handle != 0:
            Pmat.setTransposeNullSpace(tnullsp)

        ksp = petsc.PETSc.KSP().create(comm=pc.comm)
        ksp.incrementTabLevel(1, parent=pc)
        ksp.setOperators(Pmat)
        ksp.setOptionsPrefix(options_prefix)
        ksp.setFromOptions()
        ksp.setUp()
        self.ksp = ksp

    def update(self, pc):
        pass

    def apply(self, pc, X, Y):
        self.ksp.solve(X, Y)


#testing the preconditioner
parameters = {
    "ksp_type": "preonly",
    "pc_python_type": __name__ + ".pHSS_PC" #instead of pc_type
}

n = 10
k = 1
epsilon = 1

solver, w = build_problem(n, parameters, k, epsilon)
solver.solve()

sigma, u = w.split()
collection = tripcolor(u, cmap='coolwarm')
plt.colorbar(collection)
plt.show()