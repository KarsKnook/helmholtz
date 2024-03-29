import firedrake as fd
from firedrake.petsc import PETSc
from firedrake import dmhooks
import warnings
warnings.simplefilter('ignore')  # to suppress the ComplexWarning in errornorm 


def build_problem_sin2(mesh_refinement, parameters, k, delta, delta_0=0):
    """
    Build problem for u = sin^2(pi*x)sin^2(pi*y) on UnitSquareMesh
    delta is for shift preconditioning
    delta_0 is a problem parameter
    """
    mesh = fd.UnitSquareMesh(mesh_refinement, mesh_refinement)

    V = fd.VectorFunctionSpace(mesh, "DG", degree=1, dim=2)
    Q = fd.FunctionSpace(mesh, "CG", degree=2)
    W = V * Q

    sigma, u = fd.TrialFunctions(W)
    tau, v = fd.TestFunctions(W)

    # RHS for u = sin^2(pi*x)sin^2(pi*y)
    x, y = fd.SpatialCoordinate(mesh)
    f = ((-delta_0+1j*k)**2*fd.sin(fd.pi*x)**2*fd.sin(fd.pi*y)**2
         + fd.pi**2*(fd.cos(2*fd.pi*(x+y)) + fd.cos(2*fd.pi*(x-y)) - fd.cos(2*fd.pi*x) - fd.cos(2*fd.pi*y)))

    # linear variational form of original problem
    a = (fd.inner(fd.Constant(delta_0-1j*k)*sigma, tau)*fd.dx
         - fd.inner(fd.grad(u), tau)*fd.dx
         + fd.inner(sigma, fd.grad(v))*fd.dx
         + fd.inner(fd.Constant(delta_0-1j*k)*u, v)*fd.dx
         + fd.inner(u, v)*fd.ds)
    L = - fd.inner(f/fd.Constant(-delta_0+1j*k), v)*fd.dx

    # setting up a linear variational solver and passing in k, delta and f in appctx
    appctx = {"k": k, "delta": delta, "f": f}
    w = fd.Function(W)
    vpb = fd.LinearVariationalProblem(a, L, w)
    solver = fd.LinearVariationalSolver(vpb, solver_parameters=parameters, appctx=appctx)

    return solver, w


def build_problem_point_source(mesh_refinement, parameters, k, delta, delta_0=0):
    """
    Point source on UnitDiskMesh
    delta is for shift preconditioning
    delta_0 is a problem parameter
    """
    mesh = fd.UnitDiskMesh(mesh_refinement)

    V = fd.VectorFunctionSpace(mesh, "DG", degree=1, dim=2)
    Q = fd.FunctionSpace(mesh, "CG", degree=2)
    W = V * Q

    sigma, u = fd.TrialFunctions(W)
    tau, v = fd.TestFunctions(W)

    # RHS for u = sin^2(pi*x)sin^2(pi*y)
    x, y = fd.SpatialCoordinate(mesh)
    f = fd.conditional(fd.le(x**2+y**2, 0.01), 1, 0)

    # linear variational form of original problem
    a = (fd.inner(fd.Constant(delta_0-1j*k)*sigma, tau)*fd.dx
         - fd.inner(fd.grad(u), tau)*fd.dx
         + fd.inner(sigma, fd.grad(v))*fd.dx
         + fd.inner(fd.Constant(delta_0-1j*k)*u, v)*fd.dx
         + fd.inner(u, v)*fd.ds)
    L = - fd.inner(f/fd.Constant(-delta_0+1j*k), v)*fd.dx

    # setting up a linear variational solver and passing in k, delta and f in appctx
    appctx = {"k": k, "delta": delta, "f": f}
    w = fd.Function(W)
    vpb = fd.LinearVariationalProblem(a, L, w)
    solver = fd.LinearVariationalSolver(vpb, solver_parameters=parameters, appctx=appctx)

    return solver, w


def build_problem_box_source(mesh_refinement, parameters, k, delta, delta_0=0):
    """
    Source in 0.2x0.2 box on UnitSquareMesh
    delta is for shift preconditioning
    delta_0 is a problem parameter
    """
    mesh = fd.UnitSquareMesh(mesh_refinement, mesh_refinement)

    V = fd.VectorFunctionSpace(mesh, "DG", degree=1, dim=2)
    Q = fd.FunctionSpace(mesh, "CG", degree=2)
    W = V * Q

    sigma, u = fd.TrialFunctions(W)
    tau, v = fd.TestFunctions(W)

    # RHS for u = sin^2(pi*x)sin^2(pi*y)
    x, y = fd.SpatialCoordinate(mesh)
    f = (fd.conditional(fd.ge(x, 0.4), 1, 0)
         *fd.conditional(fd.le(x, 0.6), 1, 0)
         *fd.conditional(fd.ge(y, 0.4), 1, 0)
         *fd.conditional(fd.le(y, 0.6), 1, 0))

    # linear variational form of original problem
    a = (fd.inner(fd.Constant(delta_0-1j*k)*sigma, tau)*fd.dx
         - fd.inner(fd.grad(u), tau)*fd.dx
         + fd.inner(sigma, fd.grad(v))*fd.dx
         + fd.inner(fd.Constant(delta_0-1j*k)*u, v)*fd.dx
         + fd.inner(u, v)*fd.ds)
    L = - fd.inner(f/fd.Constant(-delta_0+1j*k), v)*fd.dx

    # setting up a linear variational solver and passing in k, delta and f in appctx
    appctx = {"k": k, "delta": delta, "f": f}
    w = fd.Function(W)
    vpb = fd.LinearVariationalProblem(a, L, w)
    solver = fd.LinearVariationalSolver(vpb, solver_parameters=parameters, appctx=appctx)

    return solver, w


def build_problem_5_2(mesh_refinement, parameters, k, delta, delta_0=0):
    """
    f = 1 on UnitSquareMesh
    delta is for shift preconditioning
    delta_0 is a problem parameter
    """
    mesh = fd.UnitSquareMesh(mesh_refinement, mesh_refinement)

    V = fd.VectorFunctionSpace(mesh, "DG", degree=1, dim=2)
    Q = fd.FunctionSpace(mesh, "CG", degree=2)
    W = V * Q

    sigma, u = fd.TrialFunctions(W)
    tau, v = fd.TestFunctions(W)

    # RHS for f = 1
    f = fd.Constant(1, mesh)

    # linear variational form of original problem
    a = (fd.inner(fd.Constant(delta_0-1j*k)*sigma, tau)*fd.dx
         - fd.inner(fd.grad(u), tau)*fd.dx
         + fd.inner(sigma, fd.grad(v))*fd.dx
         + fd.inner(fd.Constant(delta_0-1j*k)*u, v)*fd.dx
         + fd.inner(u, v)*fd.ds)
    L = - fd.inner(f/fd.Constant(-delta_0+1j*k), v)*fd.dx

    # setting up a linear variational solver and passing in k, delta and f in appctx
    appctx = {"k": k, "delta": delta, "f": f}
    w = fd.Function(W)
    vpb = fd.LinearVariationalProblem(a, L, w)
    solver = fd.LinearVariationalSolver(vpb, solver_parameters=parameters, appctx=appctx)

    return solver, w


class pHSS_PC(fd.preconditioners.base.PCBase):
    """
    pHSS preconditioner for indefinite helmholtz equation
    Copied from firedrake/firedrake/preconditioners/massinv.py
    """
    needs_python_pmat = True

    def initialize(self, pc):
        # obtaining solver options and context
        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + "helmhss_"
        _, P = pc.getOperators()
        context = P.getPythonContext()
        self.context = context

        # obtaining k, delta and f
        k = context.appctx.get("k")
        self.k = k
        delta = context.appctx.get("delta")
        self.delta = delta
        f = context.appctx.get("f")

        # initiliazing test and trial functions
        test, trial = context.a.arguments()

        if test.function_space() != trial.function_space():
            raise ValueError("pHSS_PC only makes sense if test and trial space are the same")

        W = test.function_space()

        sigma_new, u_new = fd.TrialFunctions(W)
        tau, v = fd.TestFunctions(W)

        # LHS of coupled pHSS iteration
        a = (fd.inner(fd.Constant((delta-1j)*k)*sigma_new, tau)*fd.dx
             - fd.inner(fd.grad(u_new), tau)*fd.dx
             + fd.inner(sigma_new, fd.grad(v))*fd.dx
             + fd.inner(fd.Constant((delta-1j)*k)*u_new, v)*fd.dx
             + fd.inner(fd.Constant(k)*u_new, v)*fd.ds)

        # initializing pHSS KSP
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
        dm = W.dm
        
        ksp = PETSc.KSP().create(comm=pc.comm)
        ksp.setDM(dm)
        ksp.setDMActive(False)
        ksp.incrementTabLevel(1, parent=pc)
        ksp.setOperators(Pmat)
        ksp.setOptionsPrefix(options_prefix)
        with dmhooks.add_hooks(dm, self, appctx=context.appctx, save=False):
            ksp.setFromOptions()  # ensures appctx is passed on to the next ksp + pc
        ksp.setUp()
        self.ksp = ksp

        # initializing self.hss_rhs multiple iterations
        self.w = fd.Function(W)  # to store solution every pHSS iteration
        self.q = fd.Function(W)  # to assemble self.hss_rhs into
        self.its = PETSc.Options().getInt(options_prefix + "its")

        w_old = fd.Function(W)
        self.w_old = w_old
        sigma_old, u_old = w_old.split()
        self.hss_rhs = (fd.Constant((k-1)/(k+1))
                        *(fd.inner(fd.Constant((delta+1j)*k)*sigma_old, tau)*fd.dx
                          + fd.inner(fd.grad(u_old), tau)*fd.dx
                          - fd.inner(sigma_old, fd.grad(v))*fd.dx
                          + fd.inner(fd.Constant((delta+1j)*k)*u_old, v)*fd.dx
                          + fd.inner(fd.Constant(k)*u_old, v)*fd.ds))

    def update(self, pc):
        pass

    def apply(self, pc, X, Y):
        """
        Copied from asQ/asQ/diag_preconditioner.py
        """
        k = fd.Constant(self.k)
        #first solve
        with self.w.dat.vec_wo as w_:
            self.ksp.solve(X, w_)  # b = inner(f, v) is the only RHS term because x^0 = 0

        #all other solves
        for i in range(self.its - 1):
            self.w_old.assign(self.w)
            fd.assemble(self.hss_rhs, form_compiler_parameters=self.context.fc_params, tensor=self.q)

            with self.w.dat.vec_wo as w_, self.q.dat.vec_ro as q_:
                self.ksp.solve(q_ + 2*k/(k+1)*X, w_)  # corresponds to self.hss_rhs + inner(f,v)
        
        #copy the result into Y
        with self.w.dat.vec_ro as w_:
            w_.copy(Y)

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError
    
    def view(self, pc, viewer=None):
        super(pHSS_PC, self).view(pc, viewer)
        viewer.printfASCII("pHSS preconditioner for indefinite helmholtz equation")
        self.ksp.view(viewer)


class Schur(fd.AuxiliaryOperatorPC):
    """
    Defining the exact Schur complement for the pHSS iteration
    Copied from firedrake/demos/saddle_point_pc
    """
    def form(self, pc, v, u):
        k = self.get_appctx(pc).get("k")
        delta = self.get_appctx(pc).get("delta")
        a = (fd.inner(fd.Constant(1/((delta-1j)*k))*fd.grad(u), fd.grad(v))*fd.dx
            + fd.inner(fd.Constant((delta-1j)*k)*u, v)*fd.dx
            + fd.inner(fd.Constant(k)*u, v)*fd.ds)
        bcs = None
        return (a, bcs)