import firedrake as fd
from firedrake.petsc import PETSc
import matplotlib.pyplot as plt


def build_problem(mesh_size, parameters, k, epsilon):
    """
    Shifted problem
    """
    mesh = fd.UnitSquareMesh(mesh_size, mesh_size)

    V = fd.VectorFunctionSpace(mesh, "DG", degree=1, dim=2)
    Q = fd.FunctionSpace(mesh, "CG", degree=2)
    W = V * Q

    sigma, u = fd.TrialFunctions(W)
    tau, v = fd.TestFunctions(W)

    x, y = fd.SpatialCoordinate(mesh)
    f = ((-epsilon+1j*k)**2*fd.sin(fd.pi*x)**2*fd.sin(fd.pi*y)**2
         + fd.pi**2*(fd.cos(2*fd.pi*(x+y)) + fd.cos(2*fd.pi*(x-y)) - fd.cos(2*fd.pi*x) - fd.cos(2*fd.pi*y)))

    a = (fd.inner(fd.Constant(epsilon-1j*k)*sigma, tau)*fd.dx
         - fd.inner(fd.grad(u), tau)*fd.dx
         + fd.inner(sigma, fd.grad(v))*fd.dx
         + fd.inner(fd.Constant(epsilon-1j*k)*u, v)*fd.dx
         + fd.inner(u, v)*fd.ds)
    L = - fd.inner(f/fd.Constant(-epsilon+1j*k), v)*fd.dx

    appctx = {"k": k, "epsilon": epsilon}
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
        epsilon = context.appctx.get("epsilon", 1.0)

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
epsilon = 1

parameters = {
    "ksp_type": "gmres",
    "pc_type": "python",
    "pc_python_type": __name__ + ".pHSS_PC",
    "helmhss_ksp_type": "preonly",
    "helmhss_pc_type": "fieldsplit",
    "helmhss_pc_fieldsplit_type": "schur",
    "helmhss_pc_fieldsplit_schur_fact_type": "full",
    "helmhss_fieldsplit_0_ksp_type": "preonly",
    "helmhss_fieldsplit_0_pc_type": "ilu",
    "helmhss_fieldsplit_1_ksp_type": "preonly",
    "helmhss_fieldsplit_1_pc_type": "lu",
    "helmhss_mat_type": "nest",
    "mat_type": "matfree",
    "ksp_monitor": None,
    "ksp_converged_reason": None,
    #"ksp_view": None,
}

"""parameters = {
    "ksp_type": "preonly",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "ilu",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "lu",
    "ksp_monitor": None,
    "ksp_converged_reason": None,
    #"ksp_view": None
}"""

"""parameters = {
    "ksp_type": "gmres",
    "pc_type": "none",
    "ksp_monitor": None,
    "ksp_converged_reason": None,
    #"ksp_view": None
}"""

solver, w = build_problem(n, parameters, k, epsilon)
solver.solve()

sigma, u = w.split()
collection = fd.tripcolor(u, cmap='coolwarm')
plt.colorbar(collection)
plt.show()