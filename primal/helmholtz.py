import firedrake as fd
from firedrake.petsc import PETSc
from firedrake import dmhooks


class HSS_PC(fd.preconditioners.base.PCBase):
    """
    HSS preconditioner for indefinite helmholtz equation
    Based on firedrake/firedrake/preconditioners/massinv.py
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
            raise ValueError("HSS_PC only makes sense if test and trial space are the same")

        Q = test.function_space()

        u_new = fd.TrialFunction(Q)
        v = fd.TestFunction(Q)

        # LHS of coupled pHSS iteration
        a = (fd.inner(fd.Constant(-2j*delta*k**2 - k**2 + delta**2)*u_new, v)*fd.dx
             + fd.inner(fd.Constant(-1j*k**2 + delta)*u_new, v)*fd.ds
             + fd.inner(fd.grad(u_new), fd.grad(v))*fd.dx)

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
        dm = Q.dm
        
        ksp = PETSc.KSP().create(comm=pc.comm)
        ksp.setDM(dm)
        ksp.setDMActive(False)
        ksp.incrementTabLevel(1, parent=pc)
        ksp.setOperators(Pmat)
        ksp.setOptionsPrefix(options_prefix)
        with dmhooks.add_hooks(dm, self, appctx=context.appctx, save=False):
            ksp.setFromOptions()  # ensures appctx is passed on to the next ksp and pc
        ksp.setUp()
        self.ksp = ksp

        # initializing self.hss_rhs for multiple iterations
        self.w = fd.Function(Q)  # to store solution every HSS iteration
        self.q = fd.Function(Q)  # to assemble self.hss_rhs into
        self.its = PETSc.Options().getInt(options_prefix + "its")

        u_old = fd.Function(Q)
        self.u_old = u_old
        self.hss_rhs = (fd.Constant((k-1)/(k+1))*(fd.inner(fd.Constant(-2j*delta*k**2 + (k**2- delta**2))*u_old, v)*fd.dx)
                                                  + fd.inner(fd.Constant(-1j*k**2 - delta)*u_old, v)*fd.ds
                                                  - fd.inner(fd.grad(u_old), fd.grad(v))*fd.dx)

    def update(self, pc):
        pass

    def apply(self, pc, X, Y):
        """
        Based on asQ/asQ/diag_preconditioner.py
        """
        k = fd.Constant(self.k)
        #first solve
        with self.w.dat.vec_wo as w_:
            self.ksp.solve(X, w_)  # b = inner(f, v) is the only RHS term because x^0 = 0

        #all other solves
        for i in range(self.its - 1):
            self.u_old.assign(self.w)
            fd.assemble(self.hss_rhs, form_compiler_parameters=self.context.fc_params, tensor=self.q)

            with self.w.dat.vec_wo as w_, self.q.dat.vec_ro as q_:
                self.ksp.solve(q_ + 2*k/(k+1)*X, w_)  # corresponds to self.hss_rhs + inner(f,v)
        
        #copy the result into Y
        with self.w.dat.vec_ro as w_:
            w_.copy(Y)

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError
    
    def view(self, pc, viewer=None):
        super(HSS_PC, self).view(pc, viewer)
        viewer.printfASCII("HSS preconditioner for the indefinite helmholtz equation")
        self.ksp.view(viewer)