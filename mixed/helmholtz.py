import firedrake as fd
from firedrake.petsc import PETSc
from firedrake import dmhooks


class HSS_PC(fd.preconditioners.base.PCBase):
    """
    HSS preconditioner for mixed formulation of the indefinite Helmholtz equation
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
        bcs = context.bcs
        dm = pc.getDM()
        appctx = dmhooks.get_appctx(dm).appctx

        # obtaining k, delta and f
        k = appctx.get("k")
        self.k = k
        delta = appctx.get("delta")
        self.delta = delta
        f = appctx.get("f")

        # initiliazing test and trial functions
        test, trial = context.a.arguments()

        if test.function_space() != trial.function_space():
            raise ValueError("pHSS_PC only makes sense if test and trial space are the same")

        W = test.function_space()

        sigma_new, u_new = fd.TrialFunctions(W)
        tau, v = fd.TestFunctions(W)

        # LHS of coupled pHSS iteration
        a = (fd.inner(fd.Constant((delta - 1j)*k)*sigma_new, tau)*fd.dx
             - fd.inner(fd.grad(u_new), tau)*fd.dx
             + fd.inner(sigma_new, fd.grad(v))*fd.dx
             + fd.inner(fd.Constant((delta - 1j)*k)*u_new, v)*fd.dx
             + fd.inner(fd.Constant(k)*u_new, v)*fd.ds)

        # initializing pHSS KSP
        opts = PETSc.Options()
        mat_type = opts.getString(options_prefix + "mat_type",
                                  fd.parameters["default_matrix_type"])

        A = fd.assemble(a, bcs=bcs, form_compiler_parameters=context.fc_params,
                     mat_type=mat_type, options_prefix=options_prefix)

        Pmat = A.petscmat
        Pmat.setNullSpace(P.getNullSpace())
        tnullsp = P.getTransposeNullSpace()
        if tnullsp.handle != 0:
            Pmat.setTransposeNullSpace(tnullsp)
        
        ksp = PETSc.KSP().create(comm=pc.comm)
        ksp.setDM(dm)
        ksp.setDMActive(True)
        ksp.incrementTabLevel(1, parent=pc)
        ksp.setOperators(Pmat)
        ksp.setOptionsPrefix(options_prefix)

        fcp = self.context.fc_params
        self._ctx_ref = self.new_snes_ctx(pc, a, bcs, mat_type,
                                          fcp=fcp, options_prefix=options_prefix)
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref, save=False):
            ksp.setFromOptions()  # ensures appctx is passed on to the next ksp and pc
        self.ksp = ksp

        # initializing self.hss_rhs for multiple iterations
        self.w = fd.Function(W)  # to store solution every HSS iteration
        self.q = fd.Function(W)  # to assemble self.hss_rhs into
        self.its = PETSc.Options().getInt(options_prefix + "its", 1)

        w_old = fd.Function(W)
        self.w_old = w_old
        sigma_old, u_old = w_old.split()
        self.hss_rhs = (fd.Constant((k - 1)/(k + 1))
                        *(fd.inner(fd.Constant((delta + 1j)*k)*sigma_old, tau)*fd.dx
                          + fd.inner(fd.grad(u_old), tau)*fd.dx
                          - fd.inner(sigma_old, fd.grad(v))*fd.dx
                          + fd.inner(fd.Constant((delta + 1j)*k)*u_old, v)*fd.dx
                          + fd.inner(fd.Constant(k)*u_old, v)*fd.ds))

    def update(self, pc):
        pass

    def apply(self, pc, X, Y):
        """
        Based on asQ/asQ/diag_preconditioner.py
        """
        dm = pc.getDM()

        k = fd.Constant(self.k)
        #first solve
        self.w.assign(0)
        with self.w.dat.vec_wo as w_:
            with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
                self.ksp.solve(X, w_)  # b = inner(f, v) is the only RHS term because x^0 = 0

        #all other solves
        for i in range(self.its - 1):
            self.w_old.assign(self.w)
            # TODO we can store the assembler directly when initializing
            fd.assemble(self.hss_rhs, form_compiler_parameters=self.context.fc_params, tensor=self.q)
            
            self.w.assign(0)
            # TODO should we use the dmhooks context for both ksp solves?
            with self.w.dat.vec_wo as w_, self.q.dat.vec_ro as q_:
                q_.axpy(2*k/(k+1), X)  # corresponds to self.hss_rhs + inner(f,v)
                self.ksp.solve(q_, w_)
        
        #copy the result into Y
        with self.w.dat.vec_ro as w_:
            w_.copy(Y)

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError
    
    def view(self, pc, viewer=None):
        super(HSS_PC, self).view(pc, viewer)
        viewer.printfASCII("HSS preconditioner for the mixed formulation of the indefinite Helmholtz equation \n")
        self.ksp.view(viewer)


class Schur(fd.AuxiliaryOperatorPC):
    """
    Schur complement for the mixed HSS preconditioner
    Based on firedrake/demos/saddle_point_pc
    """
    def form(self, pc, v, u):
        k = self.get_appctx(pc).get("k")
        delta = self.get_appctx(pc).get("delta")
        a = (fd.inner(fd.Constant(1/((delta - 1j)*k))*fd.grad(u), fd.grad(v))*fd.dx
            + fd.inner(fd.Constant((delta - 1j)*k)*u, v)*fd.dx
            + fd.inner(fd.Constant(k)*u, v)*fd.ds)
        bcs = None
        return (a, bcs)
