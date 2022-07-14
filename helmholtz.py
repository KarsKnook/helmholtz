from re import L
from firedrake import *
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter('ignore')


def pHSS_iteration(sigma_old, u_old, k, epsilon, f, V, Q):
    sigma_new = TrialFunction(V)
    u_new = TrialFunction(Q)
    tau = TestFunction(V)
    v = TestFunction(Q)
    
    #solving for u_new
    a = (inner(grad(u_new), grad(v))*dx
         + inner(Constant((-epsilon+1j)**2*k**2)*u_new, v)*dx
         - inner(Constant((-epsilon+1j)*k**2)*u_new, v)*ds)
    L = (Constant((k-1)/(k+1))*(inner(Constant(-2*epsilon*k)*sigma_old, grad(v))*dx
                                - inner(grad(u_old), grad(v))*dx
                                + inner(Constant((epsilon**2 + 1)*k**2)*u_old, v)*dx
                                - inner(Constant((-epsilon+1j)*k**2)*u_old, v)*ds)
         + inner(Constant((-epsilon+1j)*2*k**2/((k+1)*(-epsilon+k*1j)))*f, v)*dx)

    uh = Function(Q)
    solve(a == L, uh, solver_parameters={"ksp_type": "preonly",
                                            "pc_type": "lu",
                                            "pc_mat_factor_solver_type": "mumps",
                                            "mat_type": "aij"})
    
    #solving for sigma_new
    a = inner(Constant((epsilon-1j)*k)*sigma_new, tau)*dx
    L = (Constant((k-1)/(k+1))*(inner(Constant((epsilon+1j)*k)*sigma_old, tau)*dx
                                + inner(grad(u_old), tau)*dx)
         + inner(grad(uh), tau)*dx)
    
    sigmah = Function(V)
    solve(a == L, sigmah, solver_parameters={"ksp_type": "preonly",
                                                "pc_type": "lu",
                                                "pc_mat_factor_solver_type": "mumps",
                                                "mat_type": "aij"})
    
    return sigmah, uh


def pHSS(sigma_0, u_0, k, epsilon, f, iters, V, Q):
    sigma = sigma_0
    u = u_0
    
    for i in range(max(1, int(iters))):
        sigma, u = pHSS_iteration(sigma, u, k, epsilon, f, V, Q)
    
    return sigma, u


def solution_plot(mesh, V, Q, k, epsilon, iters, u_exact, f_function):
    f = f_function(mesh, k, epsilon)

    #solving
    sigma_0 = interpolate(Constant((0,0), mesh), V)
    u_0 = interpolate(Constant(0, mesh), Q)
    sigma, u = pHSS(sigma_0, u_0, k, epsilon, f, iters, V, Q)

    #plotting
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5), dpi=80)

    collection0 = tripcolor(assemble(interpolate(u_exact, Q)), axes=axes[0], cmap='coolwarm')
    fig.colorbar(collection0, ax=axes[0], orientation='horizontal', fraction=0.046, pad=0.04)
    axes[0].set_aspect("equal")
    axes[0].set_title("u_exact")
    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])

    collection1 = tripcolor(u, axes=axes[1], cmap='coolwarm')
    fig.colorbar(collection1, ax=axes[1], orientation='horizontal', fraction=0.046, pad=0.04)
    axes[1].set_aspect("equal")
    axes[1].set_title("u")
    axes[1].set_xticklabels([])
    axes[1].set_yticklabels([])

    collection2 = tripcolor(assemble(interpolate(u_exact, Q) - u), axes=axes[2], cmap='coolwarm')
    fig.colorbar(collection2, ax=axes[2], orientation='horizontal', fraction=0.046, pad=0.04)
    axes[2].set_aspect("equal")
    axes[2].set_title("difference")
    axes[2].set_xticklabels([])
    axes[2].set_yticklabels([])
    plt.show()


def error_grid_plot(mesh, V, Q, k_arr, epsilon_arr, iters, u_exact, f_function):
    errors = np.zeros((len(epsilon_arr), len(k_arr)))
    
    for i, epsilon in enumerate(epsilon_arr):
        for j, k in enumerate(k_arr): 
            f = f_function(mesh, epsilon, k)
            sigma_0 = interpolate(Constant((0,0), mesh), V)
            u_0 = interpolate(Constant(0, mesh), Q)

            sigma, u = pHSS(sigma_0, u_0, k, epsilon, f, iters, V, Q)
            errors[i, j] = errornorm(u_exact, u)
    
    plt.figure(figsize=(8, 6), dpi=80)
    plt.imshow(np.log10(errors), cmap='Reds')
    plt.xticks(np.arange(len(k_arr)), np.round(k_arr,2), rotation=90)
    plt.xlabel('k')
    plt.yticks(np.arange(len(epsilon_arr)), np.round(epsilon_arr,2))
    plt.ylabel('$\epsilon$')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title("error in order of magnitude")
    plt.show()