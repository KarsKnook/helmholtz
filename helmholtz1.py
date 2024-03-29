from firedrake import *
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter('ignore')


def pHSS_iteration1(W, f, k, epsilon, sigma_old, u_old, sigma_new, u_new, tau, v):
    """
    Perform one mixed pHSS iteration for the linear variational form of the Helmholtz equation

    Input:
        W: mixed FunctionSpace consisting out of DG(k-1)^d and CGk FunctionSpace
        f: UFL expression for RHS function f
        k: frequency, real number greater than 0
        epsilon: shift, real number greater than 0
        sigma_old: sigma^n, DG(k-1)^d Function
        u_old: u^n, CGk Function
        sigma_new: sigma^{n+1}, DG(k-1)^d TrialFunction
        u_new: u^{n+1}, CGk TrialFunction
        tau: DG(k-1)^d TestFunction
        v: CGk TestFunction
    Output:
        sigmah: sigma^{n+1}, DG(k-1)^d Function
        uh: u^{n+1}, CGk Function
    """
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

    wh = Function(W)
    solve(a == L, wh, solver_parameters={"ksp_type": "preonly",
                                         "pc_type": "lu",
                                         "pc_mat_factor_solver_type": "mumps",
                                         "mat_type": "aij"})

    return wh


def pHSS1(V, Q, f, k, epsilon, iters, sigma_0, u_0, solution=None):
    """
    Perform multiple mixed pHSS iterations for the linear variational form of the Helmholtz equation

    Input:
        V: DG(k-1)^d FunctionSpace
        Q: CGk FunctionSpace
        f: UFL expression for RHS function f
        k: frequency, real number greater than 0
        epsilon: shift, real number greater than 0
        iters: amount of iterations
        sigma_0: initial guess for sigma, DG(k-1)^d Function
        u_0: initial guess for u, CGk Function
        solution: if provided, the relative error to this solution is computed every iteration
    Output:
        (sigmah, uh): (sigma^{iters}, u^{iters}) tuple, DG(k-1)^d and CGk Functions
        error: numpy array storing relative errors over the iterations
    """
    sigma = sigma_0
    u = u_0

    W = V * Q
    sigma_new, u_new = TrialFunctions(W)
    tau, v = TestFunctions(W)

    if solution != None:
        error = np.zeros((1, iters))
    
    for i in range(iters):
        wh = pHSS_iteration1(W, f, k, epsilon, sigma, u, sigma_new, u_new, tau, v)
        sigma, u = split(wh)

        if solution != None:
            uh = wh.split()[1]
            error[0, i] = errornorm(solution, uh)

    if solution != None:
        return wh.split()[0], wh.split()[1], error/norm(solution)
    return wh.split()


def direct_solver1(V, Q, f, k, epsilon):
    """
    Directly solve the linear variational form of the Helmholtz equation

    Input:
        V: DG(k-1)^d FunctionSpace
        Q: CGk FunctionSpace
        f: UFL expression for RHS function f
        k: frequency, real number greater than 0
        epsilon: shift, real number greater than 0
    Output:
        sigmah: numerical solution, DG(k-1)^d Function
        uh: numerical solution, CGk Function
    """
    W = V * Q
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    a = (inner(Constant(epsilon - 1j*k)*sigma, tau)*dx
         - inner(grad(u), tau)*dx
         + inner(sigma, grad(v))*dx
         + inner(Constant(epsilon - 1j*k)*u, v)*dx
         + inner(u, v)*ds)
    L = - inner(f/Constant(-epsilon + 1j*k), v)*dx

    wh = Function(W)
    solve(a == L, wh, solver_parameters={"ksp_type": "preonly", 
                                         "pc_type": "lu",
                                         "pc_mat_factor_solver_type": "mumps",
                                         "mat_type": "aij"})
    
    return wh.split()


def solution_plot1(mesh, V, Q, f, k, epsilon, iters, u_exact=None, im=False):
    """
    Plot numerical solutions of direct solver and pHSS algorithm, and the difference

    Input:
        mesh: Mesh object
        V: DG(k-1)^d FunctionSpace
        Q: CGk FunctionSpace
        f: UFL expression for RHS function f
        k: frequency, real number greater than 0
        epsilon: shift, real number greater than 0
        iters: amount of pHSS iterations
        u_exact: if provided plots exact solution instead of direct solver solution
        im: if True both real and imaginary parts are plotted
    """
    # solving
    sigma_0 = interpolate(Constant((0,0), mesh), V)
    u_0 = interpolate(Constant(0, mesh), Q)
    sigma, u = pHSS1(V, Q, f, k, epsilon, iters, sigma_0, u_0)

    # plotting
    if im: # also plotting imaginary component
        fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(15, 8), dpi=80)

        if u_exact == None: # if no exact solution is provided
            sigmah, uh = direct_solver1(V, Q, f, k, epsilon)

            collection0 = tripcolor(assemble(real(uh)), axes=axes[0, 0], cmap='coolwarm')
            axes[0, 0].set_title("real part of direct solver solution")
            collection2 = tripcolor(assemble(real(uh - u)), axes=axes[0, 2], cmap='coolwarm')

            collection3 = tripcolor(assemble(imag(uh)), axes=axes[1, 0], cmap='coolwarm')
            axes[1, 0].set_title("imaginary part of direct solver solution")
            collection5 = tripcolor(assemble(imag(uh - u)), axes=axes[1, 2], cmap='coolwarm')
        else: # if an exact solution is provided
            collection0 = tripcolor(assemble(real(interpolate(u_exact, Q))), axes=axes[0, 0], cmap='coolwarm')
            axes[0, 0].set_title("real part of exact solution")
            collection2 = tripcolor(assemble(real(interpolate(real(u_exact), Q) - u)), axes=axes[0, 2], cmap='coolwarm')

            collection3 = tripcolor(assemble(imag(interpolate(u_exact, Q))), axes=axes[1, 0], cmap='coolwarm')
            axes[1, 0].set_title("imaginary part of exact solution")
            collection5 = tripcolor(assemble(imag(interpolate(u_exact, Q) - u)), axes=axes[1, 2], cmap='coolwarm')
        
        # plot 0
        fig.colorbar(collection0, ax=axes[0, 0], orientation='horizontal', fraction=0.046, pad=0.04)
        axes[0, 0].set_aspect("equal")
        axes[0, 0].set_xticklabels([])
        axes[0, 0].set_yticklabels([])

        # plot 1
        collection1 = tripcolor(assemble(real(u)), axes=axes[0, 1], cmap='coolwarm')
        fig.colorbar(collection1, ax=axes[0, 1], orientation='horizontal', fraction=0.046, pad=0.04)
        axes[0, 1].set_aspect("equal")
        axes[0, 1].set_title("real part of pHSS solution")
        axes[0, 1].set_xticklabels([])
        axes[0, 1].set_yticklabels([])

        # plot 2
        fig.colorbar(collection2, ax=axes[0, 2], orientation='horizontal', fraction=0.046, pad=0.04)
        axes[0, 2].set_aspect("equal")
        axes[0, 2].set_title("difference")
        axes[0, 2].set_xticklabels([])
        axes[0, 2].set_yticklabels([])

        # plot 3
        fig.colorbar(collection3, ax=axes[1, 0], orientation='horizontal', fraction=0.046, pad=0.04)
        axes[1, 0].set_aspect("equal")
        axes[1, 0].set_xticklabels([])
        axes[1, 0].set_yticklabels([])

        # plot 4
        collection4 = tripcolor(assemble(imag(u)), axes=axes[1, 1], cmap='coolwarm')
        fig.colorbar(collection4, ax=axes[1, 1], orientation='horizontal', fraction=0.046, pad=0.04)
        axes[1, 1].set_aspect("equal")
        axes[1, 1].set_title("imaginary part of pHSS solution")
        axes[1, 1].set_xticklabels([])
        axes[1, 1].set_yticklabels([])

        # plot 5
        fig.colorbar(collection5, ax=axes[1, 2], orientation='horizontal', fraction=0.046, pad=0.04)
        axes[1, 2].set_aspect("equal")
        axes[1, 2].set_title("difference")
        axes[1, 2].set_xticklabels([])
        axes[1, 2].set_yticklabels([])
        plt.show()
    else: # plotting only real component
        fig, axes = plt.subplots(ncols=3, figsize=(15, 5), dpi=80)

        if u_exact == None: # if no exact solution is provided
            sigmah, uh = direct_solver1(V, Q, f, k, epsilon)
            collection0 = tripcolor(uh, axes=axes[0], cmap='coolwarm')
            axes[0].set_title("direct solver solution")
            collection2 = tripcolor(assemble(uh - u), axes=axes[2], cmap='coolwarm')
        else: # if an exact solution is provided
            collection0 = tripcolor(assemble(interpolate(u_exact, Q)), axes=axes[0], cmap='coolwarm')
            axes[0].set_title("exact solution")
            collection2 = tripcolor(assemble(interpolate(u_exact, Q) - u), axes=axes[2], cmap='coolwarm')
        
        # plot 0
        fig.colorbar(collection0, ax=axes[0], orientation='horizontal', fraction=0.046, pad=0.04)
        axes[0].set_aspect("equal")
        axes[0].set_xticklabels([])
        axes[0].set_yticklabels([])

        # plot 1
        collection1 = tripcolor(u, axes=axes[1], cmap='coolwarm')
        fig.colorbar(collection1, ax=axes[1], orientation='horizontal', fraction=0.046, pad=0.04)
        axes[1].set_aspect("equal")
        axes[1].set_title("pHSS solution")
        axes[1].set_xticklabels([])
        axes[1].set_yticklabels([])

        # plot 2
        fig.colorbar(collection2, ax=axes[2], orientation='horizontal', fraction=0.046, pad=0.04)
        axes[2].set_aspect("equal")
        axes[2].set_title("difference")
        axes[2].set_xticklabels([])
        axes[2].set_yticklabels([])
        plt.show()


def error_grid_plot1(mesh, V, Q, f_function, k_arr, epsilon_arr, iters, u_exact=None):
    """
    Plot relative error wrt direct solver solution for different values of k and epsilon

    Input:
        mesh: Mesh object
        V: DG(k-1)^d FunctionSpace
        Q: CGk FunctionSpace
        f_function: function returning UFL expression for RHS function f
        k_arr: array of frequencies k
        epsilon_arr: array of shifts epsilon
        iters: amount of pHSS iterations
        u_exact: if provided plots relative error wrt exact solution
    """
    errors = np.zeros((len(epsilon_arr), len(k_arr)))
    
    for i, epsilon in enumerate(epsilon_arr):
        for j, k in enumerate(k_arr):
            # solving for specific k and epsilon
            f = f_function(mesh, k, epsilon)
            sigma_0 = interpolate(Constant((0,0), mesh), V)
            u_0 = interpolate(Constant(0, mesh), Q)
            sigma, u = pHSS1(V, Q, f, k, epsilon, iters, sigma_0, u_0)

            if u_exact == None: # if no exact solution is provided
                sigmah, uh = direct_solver1(V, Q, f, k, epsilon)
                errors[i, j] = errornorm(uh, u)/norm(uh)
            else: # if exact solution is provided
                errors[i, j] = errornorm(u_exact, u)/norm(u_exact)
    
    # plotting
    plt.figure(figsize=(8, 6), dpi=80)
    plt.imshow(np.log10(errors), cmap='Reds')
    plt.xticks(np.arange(len(k_arr)), np.round(k_arr,2), rotation=90)
    plt.xlabel('k')
    plt.yticks(np.arange(len(epsilon_arr)), np.round(epsilon_arr,2))
    plt.ylabel('$\epsilon$')
    plt.gca().invert_yaxis()
    plt.colorbar()
    if u_exact == None:
        plt.title("Relative error in order of magnitude wrt direct solver solution")
    else:
        plt.title("Relative error in order of magnitude wrt exact solution")
    plt.show()


def error_plot1(mesh, V, Q, f_function, k_arr, epsilon_arr, iters, u_exact=None):
    """
    Plot relative error wrt direct solver solution per pHSS iteration for different values of k and epsilon

    Input:
        mesh: Mesh object
        V: DG(k-1)^d FunctionSpace
        Q: CGk FunctionSpace
        f_function: function returning UFL expression for RHS function f
        k_arr: array of frequencies k
        epsilon_arr: array of shifts epsilon
        iters: amount of pHSS iterations
        u_exact: if provided plots relative error wrt exact solution
    """
    errors = np.zeros((len(k_arr), iters))

    for i in range(len(k_arr)):
        # solving
        f = f_function(mesh, k_arr[i], epsilon_arr[i])
        sigma_0 = interpolate(Constant((0,0), mesh), V)
        u_0 = interpolate(Constant(0, mesh), Q)

        if u_exact == None: # if no exact solution is provided
            sigmah, uh = direct_solver1(V, Q, f, k_arr[i], epsilon_arr[i])
            sigma, u, error = pHSS1(V, Q, f, k_arr[i], epsilon_arr[i], iters, sigma_0, u_0, solution=uh)
        else: # if an exact solution is provided
            sigma, u, error = pHSS1(V, Q, f, k_arr[i], epsilon_arr[i], iters, sigma_0, u_0, solution=u_exact)
        errors[i, :] = error

    # plotting
    plt.plot(range(1, iters+1), errors.T)
    plt.xlabel("iterations")
    plt.ylabel("relative error")
    plt.yscale("log")
    plt.legend([f"({k}, {epsilon})" for k, epsilon in zip(k_arr, epsilon_arr)])
    if u_exact == None:
        plt.title("Relative error wrt direct solver solution per pHSS iteration for different $(k, \epsilon)$")
    else:
        plt.title("Relative error wrt exact solution per pHSS iteration for different $(k, \epsilon)$")
    plt.show()