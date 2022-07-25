import firedrake as fd
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc


def build_problem(mesh_size, parameters, k):
    mesh = fd.UnitSquareMesh(mesh_size, mesh_size)

    V = fd.VectorFunctionSpace(mesh, "DG", degree=1, dim=2)
    Q = fd.FunctionSpace(mesh, "CG", degree=2)
    W = V * Q

    sigma_old = fd.interpolate(fd.Constant((0,0), mesh), V)
    u_old = fd.interpolate(fd.Constant(0, mesh), Q)

    sigma_new, u_new = fd.TrialFunctions(W)
    tau, v = fd.TestFunctions(W)

    x, y = fd.SpatialCoordinate(mesh)
    f = (-k**2*fd.sin(fd.pi*x)**2*fd.sin(fd.pi*y)**2
         + fd.pi**2*(fd.cos(2*fd.pi*(x+y)) + fd.cos(2*fd.pi*(x-y)) - fd.cos(2*fd.pi*x) - fd.cos(2*fd.pi*y)))

    a = fd.Constant(-1j*(k+1)/2)*(fd.inner(fd.Constant(-1j*k)*sigma_new, tau)*fd.dx
                                                - fd.inner(fd.grad(u_new), tau)*fd.dx
                                                + fd.inner(sigma_new, fd.grad(v))*fd.dx
                                                + fd.inner(fd.Constant(-1j*k)*u_new, v)*fd.dx
                                                + fd.inner(fd.Constant(k)*u_new,v)*fd.ds)

    L = (fd.Constant((1-k)/2)*(fd.inner(fd.Constant(1j*k)*sigma_old, tau)*fd.dx
                               + fd.inner(fd.grad(u_old), tau)*fd.dx
                               - fd.inner(sigma_old, fd.grad(v))*fd.dx
                               + fd.inner(fd.Constant(1j*k)*u_old, v)*fd.dx
                               + fd.inner(fd.Constant(k)*u_old, v)*fd.ds)
        + fd.inner(f, v)*fd.dx)

    appctx = {"k": k}
    w = fd.Function(W)
    vpb = fd.LinearVariationalProblem(a, L, w)
    solver = fd.LinearVariationalSolver(vpb, solver_parameters=parameters, appctx=appctx)

    return solver, w


class pHSS_PC(fd.preconditioners.base.PCBase):

    needs_python_pmat = True

    def initialize(self, pc):
        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + "helmhss_"
        # we assume P has things stuffed inside of it
        _, P = pc.getOperators()
        context = P.getPythonContext()

        k = context.appctx.get("k", 1.0)
        epsilon = PETSc.Options().getReal(options_prefix + "epsilon")

        test, trial = context.a.arguments()

        if test.function_space() != trial.function_space():
            raise ValueError("pHSS_PC only makes sense if test and trial space are the same")

        W = test.function_space()

        sigma_new, u_new = fd.TrialFunctions(W)
        tau, v = fd.TestFunctions(W)

        a = fd.Constant((epsilon-1j*k)*(k+1)/(2*k))*(fd.inner(fd.Constant((epsilon-1j)*k)*sigma_new, tau)*fd.dx
                                                  - fd.inner(fd.grad(u_new), tau)*fd.dx
                                                  + fd.inner(sigma_new, fd.grad(v))*fd.dx
                                                  + fd.inner(fd.Constant((epsilon-1j)*k)*u_new, v)*fd.dx
                                                  + fd.inner(fd.Constant(k)*u_new,v)*fd.ds)

        opts = PETSc.Options()
        mat_type = opts.getString(options_prefix + "mat_type",
                                  fd.parameters["default_matrix_type"])

        A = fd.assemble(a, form_compiler_parameters=context.fc_params,
                     mat_type=mat_type, options_prefix=options_prefix)

        Pmat = A.petscmat
        Pmat.setNullSpace(P.getNullSpace())
        tnullsp = P.getTransposeNullSpace()
        if tnullsp.handle != 0:
            Pmat.setTransposeNullSpace(tnullsp)

        ksp = PETSc.KSP().create(comm=pc.comm)
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

    applyTranspose = apply
    
    def view(self, pc, viewer=None):
        super(pHSS_PC, self).view(pc, viewer)
        viewer.printfASCII("pHSS preconditioner for indefinite helmholtz equation")
        self.ksp.view(viewer)


#testing the preconditioner
n = 10
k = 1

"""parameters = {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": __name__ + ".pHSS_PC",
    "helmhss_epsilon": 1,
    "mat_type": "matfree"
}"""

parameters = {
    "ksp_type": "gmres",
    "pc_type": "python",
    "pc_python_type": __name__ + ".pHSS_PC",
    "helmhss_pc_type": "fieldsplit",
    "helmhss_pc_fieldsplit_type": "schur",
    "helmhss_pc_fieldsplit_schur_fact_type": "full",
    "helmhss_fieldsplit_0_ksp_type": "preonly",
    "helmhss_fieldsplit_0_pc_type": "ilu",
    "helmhss_fieldsplit_1_ksp_type": "preonly",
    "helmhss_fieldsplit_1_pc_type": "lu",
    "helmhss_epsilon": 1,
    "mat_type": "matfree"
}

solver, w = build_problem(n, parameters, k)
solver.solve()

sigma, u = w.split()
collection = fd.tripcolor(u, cmap='coolwarm')
plt.colorbar(collection)
plt.show()