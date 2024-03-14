import firedrake as fd
import warnings
import numpy as np
warnings.simplefilter('ignore')  # to suppress the ComplexWarning in errornorm 


class CustomTransferManager(fd.TransferManager):

    def prolong(self, source, target):
        # Vc -> Vf
        try:
            super(CustomTransferManager, self).prolong(source, target)
        except NotImplementedError:
            pass

    def restrict(self, source, target):
        # Vf^* -> Vc^*
        try:
            super(CustomTransferManager, self).restrict(source, target)
        except NotImplementedError:
            pass

    def inject(self, source, target):
        # Vf-> Vc
        try:
            super(CustomTransferManager, self).inject(source, target)
        except NotImplementedError:
            pass


def helmholtz_LHS(sigma, u, tau, v, k, delta_0):
    """
    Assembles the LHS of the mixed formulation of the Helmholtz equation

    :param sigma: DG(k-1) trial function
    :param u: CGk trial function
    :param tau: DG(k-1) test function
    :param v: CGk test function
    :param k: frequency parameter
    :param delta_0: shift problem parameter
    :return a: UFL expression for the LHS
    """
    a = (fd.inner(fd.Constant(delta_0 - 1j*k)*sigma, tau)*fd.dx
         - fd.inner(fd.grad(u), tau)*fd.dx
         + fd.inner(sigma, fd.grad(v))*fd.dx
         + fd.inner(fd.Constant(delta_0-1j*k)*u, v)*fd.dx
         + fd.inner(u, v)*fd.ds)
    return a


def helmholtz_RHS(f, v, k, delta_0):
    """
    Assembles the RHS of the mixed formulation of the Helmholtz equation

    :param f: UFL expression for RHS function f
    :param v: CGk test function
    :param k: frequency parameter
    :param delta_0: shift problem parameter
    :return L: UFL expression for the RHS
    """
    return - fd.inner(f/fd.Constant(-delta_0 + 1j*k), v)*fd.dx


def build_problem(mesh, f, parameters, k, delta, delta_0, degree):
    """
    Given mesh and RHS function f, assembles the linear variational solver for the mixed formulation

    :param mesh: Mesh object
    :param f: UFL expression for RHS function f
    :param parameters: dictionary of solver parameters
    :param k: frequency parameter
    :param delta: shift preconditioning parameter
    :param delta_0: shift problem parameter
    :param degree: degree of CGk
    :return solver: solver object for linear variational problem
    :return w: solution function
    """
    # build mixed function space and define test and trial functions
    V = fd.VectorFunctionSpace(mesh, "DG", degree=degree-1, dim=2)
    Q = fd.FunctionSpace(mesh, "CG", degree=degree)
    W = V * Q
    sigma, u = fd.TrialFunctions(W)
    tau, v = fd.TestFunctions(W)

    # build LHS and RHS of mixed helmholtz problem
    a = helmholtz_LHS(sigma, u, tau, v, k, delta_0)
    L = helmholtz_RHS(f, v, k, delta_0)

    # build problem and solver for mixed helmholtz problem
    appctx = {"k": k, "delta": delta, "f": f}
    w = fd.Function(W)
    vpb = fd.LinearVariationalProblem(a, L, w)
    solver = fd.LinearVariationalSolver(vpb, solver_parameters=parameters, appctx=appctx)

    transfer = CustomTransferManager()
    solver.set_transfer_manager(transfer)
    return solver, w


def build_problem_box_source(nx, parameters, k, delta, delta_0, degree, HSS_method, levels):
    """
    Assembles linear variational solver for source function in 0.2x0.2 box on UnitSquareMesh

    :param nx: number of elements along an edge
    :param parameters: dictionary of solver parameters
    :param k: frequency parameter
    :param delta: shift preconditioning parameter
    :param delta_0: shift problem parameter
    :param degree: degree of CGk
    :param HSS_method: solver method for HSS problems
    :param levels: amount of levels in geometric mg
    :return solver: solver object for linear variational problem
    :return w: solution function
    """
    if HSS_method == "mg":
        nx = nx // 2**(levels-1)
        mesh = fd.UnitSquareMesh(nx, nx)
        hierarchy = fd.MeshHierarchy(mesh, levels-1)
        mesh = hierarchy[-1]
    else:
        mesh = fd.UnitSquareMesh(nx, nx)

    x, y = fd.SpatialCoordinate(mesh)
    f = (fd.conditional(fd.ge(x, 0.4), 1, 0)
         *fd.conditional(fd.le(x, 0.6), 1, 0)
         *fd.conditional(fd.ge(y, 0.4), 1, 0)
         *fd.conditional(fd.le(y, 0.6), 1, 0))

    return build_problem(mesh, f, parameters, k, delta, delta_0, degree)


def build_problem_uniform_source(nx, parameters, k, delta, delta_0, degree, HSS_method, levels):
    """
    Assembles linear variational solver for f = 1 on UnitSquareMesh

    :param nx: number of elements along an edge
    :param parameters: dictionary of solver parameters
    :param k: frequency parameter
    :param delta: shift preconditioning parameter
    :param delta_0: shift problem parameter
    :param degree: degree of CGk
    :param HSS_method: solver method for HSS problems
    :param levels: amount of levels in geometric mg
    :return solver: solver object for linear variational problem
    :return w: solution function
    """
    if HSS_method == "mg":
        nx = nx // 2**(levels-1)
        mesh = fd.UnitSquareMesh(nx, nx)
        hierarchy = fd.MeshHierarchy(mesh, levels-1)
        mesh = hierarchy[-1]
    else:
        mesh = fd.UnitSquareMesh(nx, nx)

    f = fd.Constant(1, mesh)

    return build_problem(mesh, f, parameters, k, delta, delta_0, degree)


def build_problem_sin2(nx, parameters, k, delta, delta_0, degree, HSS_method, levels):
    """
    Assembles linear variational solver for u = sin^2(pi*x)sin^2(pi*y) (method of 
    manufactured solutions) on UnitSquareMesh

    :param nx: number of elements along an edge
    :param parameters: dictionary of solver parameters
    :param k: frequency parameter
    :param delta: shift preconditioning parameter
    :param delta_0: shift problem parameter
    :param degree: degree of CGk
    :param HSS_method: solver method for HSS problems
    :param levels: amount of levels in geometric mg
    :return solver: solver object for linear variational problem
    :return w: solution function
    """
    if HSS_method == "mg":
        nx = nx // 2**(levels-1)
        mesh = fd.UnitSquareMesh(nx, nx)
        hierarchy = fd.MeshHierarchy(mesh, levels-1)
        mesh = hierarchy[-1]
    else:
        mesh = fd.UnitSquareMesh(nx, nx)

    x, y = fd.SpatialCoordinate(mesh)
    f = (( -delta_0 + 1j*k)**2 * fd.sin(fd.pi*x)**2 * fd.sin(fd.pi*y)**2
         + fd.pi**2*(fd.cos(2*fd.pi*(x + y)) + fd.cos(2*fd.pi*(x - y)) 
         - fd.cos(2*fd.pi*x) - fd.cos(2*fd.pi*y)))

    return build_problem(mesh, f, parameters, k, delta, delta_0, degree)