import firedrake as fd
import warnings
import numpy as np
warnings.simplefilter('ignore')  # to suppress the ComplexWarning in errornorm

def helmholtz_LHS(u, v, k, delta_0):
    """
    Assembles the LHS of the primal formulation of the Helmholtz equation

    :param u: CGk trial function
    :param v: CGk test function
    :param k: frequency parameter
    :param delta_0: shift problem parameter
    :return a: UFL expression for the LHS
    """
    a = (fd.inner(fd.Constant((-delta_0 + 1j*k)**2)*u, v)*fd.dx
        + fd.inner(fd.grad(u), fd.grad(v))*fd.dx
        - fd.inner(fd.Constant(-delta_0 + 1j*k)*u, v)*fd.ds)
    return a


def helmholtz_RHS(f, v, k, delta_0):
    """
    Assembles the RHS of the primal formulation of the Helmholtz equation

    :param f: UFL expression for RHS function f
    :param v: CGk test function
    :param k: frequency parameter
    :param delta_0: shift problem parameter
    :return L: UFL expression for the RHS
    """
    return fd.inner(f, v)*fd.dx


def build_problem(mesh, f, parameters, k, delta, delta_0, degree):
    """
    Given mesh and RHS function f, assembles the linear variational solver for the primal formulation

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
    # build function space and define test and trial functions
    W = fd.FunctionSpace(mesh, "CG", degree=degree)
    u = fd.TrialFunction(W)
    v = fd.TestFunction(W)

    # build LHS and RHS of the primal helmholtz problem
    a = helmholtz_LHS(u, v, k, delta_0)
    L = helmholtz_RHS(f, v, k, delta_0)

    # build problem and solver for primal helmholtz problem
    appctx = {"k": k, "delta": delta, "f": f}
    w = fd.Function(W)
    vpb = fd.LinearVariationalProblem(a, L, w)
    solver = fd.LinearVariationalSolver(vpb, solver_parameters=parameters, appctx=appctx, options_prefix="")

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
    f = ((-delta_0 + 1j*k)**2 * fd.sin(fd.pi*x)**2 * fd.sin(fd.pi*y)**2
         + fd.pi**2 * (fd.cos(2*fd.pi*(x + y)) + fd.cos(2*fd.pi*(x - y))
         - fd.cos(2*fd.pi*x) - fd.cos(2*fd.pi*y)))

    return build_problem(mesh, f, parameters, k, delta, delta_0, degree)
