###########################################################################
# Import packages
import numpy as np
import numpy.random as rnd
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance
import scipy as sp
from tqdm import tqdm

###########################################################################
# Algorithm Parameters
params = {}
params["num_samples"] = 500
params["num_iters"] = 50
params["dim_x"] = 5
params["dim_z"] = 5
params["rho"] = 2.5
params["h"] = 0.1
params["gamma"] = 0.5

# Gaussian objective
target_mean = 4*rnd.normal(size=(params["dim_x"], ))
# target_mean = np.array([0.])
params["target_mean"] = target_mean

target_covar = 7*rnd.normal(size=(params["dim_x"], params["dim_x"]))
target_covar = target_covar.T@target_covar + 0.01*np.identity(params["dim_x"])
# target_covar = np.array([1/3]).reshape(1,1)
params["target_covar"] = target_covar

grad_f = lambda x: np.linalg.inv(target_covar)@(x-target_mean)

###########################################################################
# Helper functions
def wass_dist(xs, target_mean, target_covar):
    empirical_mean = np.mean(xs.T, axis=0)
    empirical_cov  = EmpiricalCovariance(assume_centered=False).fit(xs.T).covariance_

    dist = np.linalg.norm(empirical_mean - target_mean)**2 + \
            np.trace(empirical_cov + target_covar - 2*sp.linalg.sqrtm( sp.linalg.sqrtm(target_covar)@empirical_cov@sp.linalg.sqrtm(target_covar)  ) )
    
    dist = np.sqrt(dist)

    return(dist)

###########################################################################
def run_methods(params):

    num_samples = params["num_samples"]
    num_iters = params["num_iters"]

    h = params["h"]
    rho = params["rho"]
    gamma = params["gamma"]

    dim_x = params["dim_x"]
    dim_z = params["dim_z"]

    target_mean = params["target_mean"]
    target_covar = params["target_covar"]

    METHOD = params["METHOD"]

    # Set-up samples and output
    x = rnd.normal(size=(dim_x, num_samples))
    z = np.zeros((dim_x, num_samples))
    p = np.zeros((dim_x, num_samples))
    wasserstein_dists = []
    covars = []
    means = []

    # Main Loop
    if METHOD == "ADMM":
        # Set-up primal problems
        x_var = cp.Variable(dim_x)
        z_var = cp.Variable(dim_z)

        x_par = cp.Parameter(dim_x)
        z_par = cp.Parameter(dim_z)

        dual_par = cp.Parameter(dim_x)
        noise_par = cp.Parameter(dim_x)

        if params["ADMM type"] == "g=0":
            x_objective = cp.Minimize( 0.5*cp.quad_form(x_var-target_mean, np.linalg.inv(target_covar)) \
                                    + rho/2*cp.norm( x_var - z_par + dual_par + noise_par)**2)
            z_objective = cp.Minimize( 0. \
                                    + rho/2*cp.norm( x_par - z_var + dual_par)**2)
            
        elif params["ADMM type"] == "f=g":
            x_objective = cp.Minimize( 0.25*cp.quad_form(x_var-target_mean, np.linalg.inv(target_covar)) \
                                    + rho/2*cp.norm( x_var - z_par + dual_par + noise_par)**2)
            z_objective = cp.Minimize( 0.25*cp.quad_form(z_var-target_mean, np.linalg.inv(target_covar)) \
                                    + rho/2*cp.norm( x_par - z_var + dual_par)**2)

        x_problem = cp.Problem(x_objective)
        z_problem = cp.Problem(z_objective)

        # Run algorithm
        for iter in tqdm(range(num_iters)):
            wasserstein_dists.append(wass_dist(x, target_mean, target_covar))
            # means.append(np.mean(x.T, axis=0).item())
            # covars.append(EmpiricalCovariance(assume_centered=False).fit(x.T).covariance_.item())
            for sample in range(num_samples):
                # x-update
                z_par.value = z[:, sample]
                dual_par.value = p[:, sample]
                noise_par.value = rnd.normal(size=(dim_x,))   
                x_problem.solve()
                x[:, sample] = x_var.value

                # z-update
                x_par.value = x_var.value
                z_problem.solve()
                z[:, sample] = z_var.value

                # dual update
                p[:, sample] += (x_var.value-z_var.value)

    elif METHOD == "LGD":
        for iter in tqdm(range(num_iters)):
            wasserstein_dists.append(wass_dist(x, target_mean, target_covar))
            for sample in range(num_samples):
                x[:, sample] += -h*grad_f(x[:, sample]) + np.sqrt(2*h)*rnd.normal(size=(dim_x,))

    elif METHOD == "PLA":
        x_var = cp.Variable(dim_x)
        y_par = cp.Parameter(dim_x)

        x_objective = cp.Minimize(   0.5*cp.quad_form(x_var-target_mean, np.linalg.inv(target_covar))\
                                    + 1/(2*gamma)*cp.norm( x_var - y_par)**2)
        x_problem = cp.Problem(x_objective)
        
        for iter in tqdm(range(num_iters)):
            wasserstein_dists.append(wass_dist(x, target_mean, target_covar))
            # means.append(np.mean(x.T, axis=0).item())
            # covars.append(EmpiricalCovariance(assume_centered=False).fit(x.T).covariance_.item())
            for sample in range(num_samples):
                y_par.value = x[:, sample] + np.sqrt(2*gamma)*rnd.normal(size=(dim_x,))
                
                x_problem.solve()
                x[:, sample] = x_var.value

    elif METHOD == "PD":
        # Set-up primal problem
        x_var = cp.Variable(dim_x)
        z_var = cp.Variable(dim_z)

        dual_par = cp.Parameter(1)
        noise_par = cp.Parameter(1)

        objective = cp.Minimize( 0.5*x_var**2 \
                                    + rho/2*cp.norm( x_var - z_var + dual_par + noise_par)**2)
        problem = cp.Problem(objective)

        # Run algorithm
        for iter in tqdm(range(num_iters)):
            wasserstein_dists.append(wass_dist(x, target_mean, target_covar))
            for sample in range(num_samples):
                # primal update
                dual_par.value = p[:, sample]
                noise_par.value = rnd.normal(size=(1,))
                problem.solve()
                x[:, sample] = x_var.value
                z[:, sample] = z_var.value

                # dual update
                p[:, sample] += (x_var.value-z_var.value) 

    return wasserstein_dists, x


def cont_time_ADMM(params):
    num_samples = params["num_samples"]
    num_iters = params["num_iters"]

    rho = params["rho"]

    dim_x = params["dim_x"]
    dim_z = params["dim_z"]

    target_mean = params["target_mean"]
    target_covar = params["target_covar"]

    METHOD = params["METHOD"]

    # Set-up samples and output
    x = rnd.normal(loc=3.,size=(dim_x, num_samples, num_iters))
    wasserstein_dists = []

    # Main Loop
    for sample in tqdm(range(num_samples)):
        sol = sp.integrate.solve_ivp(ode, [0, num_iters/rho], np.array([x[0, sample, 0], 0., 0.]).reshape(-1,), t_eval=[i/rho for i in range(1, num_iters)], args=[rho])
        x[0, sample, 1:] = sol.y[0, :]
    for iter in range(num_iters):
        wasserstein_dists.append(wass_dist(x[:, :, iter], target_mean, target_covar))

    return wasserstein_dists, x[:, :, -1]

def ode(t, xzu, rho):
    state = xzu.reshape(-1,1)
    A = np.array([[-(1+rho), rho, -rho], [rho, -rho, rho], [rho, -rho, 0]])
    B = np.array([-rho*rnd.normal(), -1, 0]).reshape(3,1)

    dxdzdu = A@state + B
    return dxdzdu.reshape(-1)


###########################################################################
# Run method
params["METHOD"] = "ADMM"
params["ADMM type"] = "g=0"
# dist_cont_ADMM, x_cont_ADMM = cont_time_ADMM(params)
dist_ADMM_0, x_ADMM_0 = run_methods(params)
params["ADMM type"] = "f=g"
# dist_cont_ADMM, x_cont_ADMM = cont_time_ADMM(params)
dist_ADMM_fg, x_ADMM_fg = run_methods(params)

# params["METHOD"] = "PD"
# dist_PD, x_PD = run_methods(params)

params["METHOD"] = "LGD"
dist_LGD, x_LGD = run_methods(params)

params["METHOD"] = "PLA"
dist_PLA, x_PLA = run_methods(params)

# Plot results
# plt.hist(x_ADMM[0, :], alpha=0.4, bins=100, label="ADMM sampler")
# plt.hist(x_cont_ADMM[0, :], alpha=0.4, bins=100, label="ADMM cont")
# plt.hist(x_PLA[0, :], alpha=0.4, bins=100, color='red', label="PLA")
# plt.hist(x_PD[0, :], alpha=0.4, bins=100, color='green', label="PD")

# f = lambda x: 0.5*(x-target_mean)**2/target_covar
# xs = list(np.arange(-10, 10., 0.02))
# fs = [np.exp(-f(x.reshape(-1,1))).item() for x in xs]
# max_f = max(fs)
# fs = [10*i/max_f for i in fs]
# plt.plot(xs, fs, color="black", label=r"$\propto \exp{- (f(x) + g(x))}$")
# plt.xlabel("x")
# plt.ylabel("Number of samples")
# plt.legend()
# plt.show()

plt.plot(range(len(dist_ADMM_0)), dist_ADMM_0, label="ADMM (g=0) (ρ="+str(params["rho"])+")")
plt.plot(range(len(dist_ADMM_fg)), dist_ADMM_fg, label="ADMM (f=g) (ρ="+str(params["rho"])+")")
plt.plot(range(len(dist_PLA)), dist_PLA, label="PLA (γ="+str(params["gamma"])+")")
plt.plot(range(len(dist_LGD)), dist_LGD, label="LGD (h="+str(params["h"])+")")
plt.xlabel("Iteration")
plt.ylabel("Wasserstein distance")
plt.title("d="+str(params["dim_x"]))
plt.legend()
plt.show()


# Mean/Covar Propagation in ADMM
# r = params['rho']
# M = np.array([[0, r/(r+1), -r/(r+1), -r/(r+1)], [0, r/(r+1), 1/(r+1), -r/(r+1)], [0, 0, 0, 0], [0, 0, 0, 1]])
# S = np.array([[0, (r/(r+1))**2, (r/(r+1))**2, (r/(r+1))**2], [0, (r/(r+1))**2, (1/(r+1))**2, (r/(r+1))**2], [0, 0, 0, 0], [0, 0, 0, 1]])
# mus = np.array([3, 0, 0, 0]).reshape(-1, 1)
# covs = np.array([1, 0, 0, 1]).reshape(-1, 1)
# mu_seq = []
# covs_seq = []
# mu_seq.append(mus[0, 0])
# covs_seq.append(covs[0,0])
# for _ in range(len(dist_ADMM)-1):
#     mus = M@mus
#     covs = S@covs
#     mu_seq.append(mus[0, 0])
#     covs_seq.append(covs[0,0])
# plt.plot(range(len(means_ADMM)), means_ADMM, label="ADMM - empir mean")
# plt.plot(range(len(covars_ADMM)), covars_ADMM, label="ADMM - empir covar")
# plt.plot(range(len(mu_seq)), mu_seq, label="ADMM - theor mean")
# plt.plot(range(len(covs_seq)), covs_seq, label="ADMM - theor covar")
# plt.legend()
# plt.title("ρ="+str(r))
# plt.show()

# Mean/Covar Propagation in PLA
# gamma = params['gamma']
# mu = 3.
# mu_seq = []
# mu_seq.append(mu)
# covars_seq = []
# cov = 1.
# covars_seq.append(cov)
# for _ in range(len(means_PLA)-1):
#     mu = 1/(1+gamma)*mu
#     mu_seq.append(mu)
#     cov = (1/(1+gamma))**2*cov + (2*gamma)/(1+gamma)**2*1
#     covars_seq.append(cov)
# plt.plot(range(len(means_PLA)), means_PLA, label="PLA - empir mean")
# plt.plot(range(len(mu_seq)), mu_seq, label="PLA - theor mean")
# plt.plot(range(len(covars_PLA)), covars_PLA, label="PLA - empir covar")
# plt.plot(range(len(covars_seq)), covars_seq, label="PLA - theor covar")
# plt.legend()
# plt.title("γ="+str(gamma))
# plt.show()




