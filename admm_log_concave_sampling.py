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
params["num_samples"] = 200
params["num_iters"] = 40
params["dim_x"] = 1
params["dim_z"] = 1
params["rho"] = 2.
params["h"] = 0.1
params["gamma"] = 0.3

# Gaussian objective
target_mean = np.array([3.])
params["target_mean"] = target_mean

target_covar = np.array([4.]).reshape(1,1)
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
    p = np.zeros((1, num_samples))
    wasserstein_dists = []

    # Main Loop
    if METHOD == "ADMM":
        # Set-up primal problems
        x_var = cp.Variable(dim_x)
        z_var = cp.Variable(dim_z)

        x_par = cp.Parameter(dim_x)
        z_par = cp.Parameter(dim_z)

        dual_par = cp.Parameter(1)
        noise_par = cp.Parameter(1)

        x_objective = cp.Minimize( 0.25*cp.quad_form(x_var-target_mean, np.linalg.inv(target_covar)) \
                                    + rho/2*cp.norm( x_var - z_par + dual_par + noise_par)**2)
        x_problem = cp.Problem(x_objective)

        z_objective = cp.Minimize( 0.25*cp.quad_form(z_var-target_mean, np.linalg.inv(target_covar)) \
                                    + rho/2*cp.norm( x_par - z_var + dual_par)**2)
        z_problem = cp.Problem(z_objective)

        # Run algorithm
        for iter in tqdm(range(num_iters)):
            wasserstein_dists.append(wass_dist(x, target_mean, target_covar))
            for sample in range(num_samples):
                # x-update
                z_par.value = z[:, sample]
                dual_par.value = p[:, sample]
                noise_par.value = rnd.normal(size=(1,))
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
                x[:, sample] += -h*grad_f(x[:, sample]) + np.sqrt(2*h)*rnd.normal(size=(1,))

    elif METHOD == "PLA":
        x_var = cp.Variable(dim_x)
        y_par = cp.Parameter(dim_x)

        x_objective = cp.Minimize( 0.5*cp.quad_form(x_var-target_mean, np.linalg.inv(target_covar)) \
                                    + 1/(2*gamma)*cp.norm( x_var - y_par)**2)
        x_problem = cp.Problem(x_objective)
        
        for iter in tqdm(range(num_iters)):
            wasserstein_dists.append(wass_dist(x, target_mean, target_covar))
            for sample in range(num_samples):
                y_par.value = x[:, sample] + np.sqrt(2*gamma)*rnd.normal(size=(1,))

                x_problem.solve()
                x[:, sample] = x_var.value

    return wasserstein_dists, x

###########################################################################
# Run method
params["METHOD"] = "ADMM"
dist_ADMM, x_ADMM = run_methods(params)

params["METHOD"] = "LGD"
dist_LGD, x_LGD = run_methods(params)

params["METHOD"] = "PLA"
dist_PLA, x_PLA = run_methods(params)

# Plot results
plt.hist(x_ADMM[0, :], alpha=0.4, bins=100)

f = lambda x: 0.5*(x-target_mean)*np.linalg.inv(target_covar)*(x-target_mean)
xs = list(np.arange(-4, 9., 0.02))
fs = [np.exp(-f(x.reshape(-1,1))).item() for x in xs]
max_f = max(fs)
fs = [5*i/max_f for i in fs]
plt.plot(xs, fs, color="black", label=r"$\propto \exp{-\sum_{i \in \mathcal{V}} f_i(x)}$")
plt.xlabel("x")
plt.ylabel("Number of samples")
plt.show()

plt.plot(range(len(dist_ADMM)), dist_ADMM, label="ADMM")
plt.plot(range(len(dist_LGD)), dist_LGD, label="LGD")
plt.plot(range(len(dist_PLA)), dist_PLA, label="PLA")
plt.legend()
plt.show()




