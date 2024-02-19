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
params["num_samples"] = 100
params["num_iters"] = 20
params["dim_x"] = 1
params["dim_z"] = 1
params["rho"] = 2.5
params["h"] = 0.1
params["gamma"] = 0.5

# Gaussian objective
target_mean = np.array([0.])
params["target_mean"] = target_mean

target_covar = np.array([1.]).reshape(1,1)
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
    x = rnd.normal(loc=3.,size=(dim_x, num_samples))
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

        x_objective = cp.Minimize( 0.5*x_var**2  \
                                    + rho/2*cp.norm( x_var - z_par + dual_par + noise_par)**2)
        x_problem = cp.Problem(x_objective)

        z_objective = cp.Minimize( 0. \
                                    + rho/2*cp.norm( x_par - z_var + dual_par)**2)
        z_problem = cp.Problem(z_objective)

        # Run algorithm
        for iter in tqdm(range(num_iters)):
            wasserstein_dists.append(wass_dist(x, target_mean, target_covar))
            # wasserstein_dists.append(np.mean(x.T, axis=0).item())
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

        x_objective = cp.Minimize(   0.5*x_var**2\
                                    + 1/(2*gamma)*cp.norm( x_var - y_par)**2)
        x_problem = cp.Problem(x_objective)
        
        for iter in tqdm(range(num_iters)):
            wasserstein_dists.append(wass_dist(x, target_mean, target_covar))
            # wasserstein_dists.append(np.mean(x.T, axis=0).item())
            for sample in range(num_samples):
                y_par.value = x[:, sample] + np.sqrt(2*gamma)*rnd.normal(size=(1,))
                
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
    B = np.array([-rho, 0, 0]).reshape(3,1)

    dxdzdu = A@state + B@rnd.normal(size=(1,1))
    return dxdzdu.reshape(-1)


###########################################################################
# Run method
params["METHOD"] = "ADMM"
dist_cont_ADMM, x_cont_ADMM = cont_time_ADMM(params)
dist_ADMM, x_ADMM = run_methods(params)

# params["METHOD"] = "PD"
# dist_PD, x_PD = run_methods(params)

# params["METHOD"] = "LGD"
# dist_LGD, x_LGD = run_methods(params)

# params["METHOD"] = "PLA"
# dist_PLA, x_PLA = run_methods(params)

# Plot results
plt.hist(x_ADMM[0, :], alpha=0.4, bins=100, label="ADMM sampler")
plt.hist(x_cont_ADMM[0, :], alpha=0.4, bins=100, label="ADMM cont")
# plt.hist(x_PLA[0, :], alpha=0.4, bins=100, color='red', label="PLA")
# plt.hist(x_PD[0, :], alpha=0.4, bins=100, color='green', label="PD")

f = lambda x: 0.5*x**2
xs = list(np.arange(-5, 12., 0.02))
fs = [np.exp(-f(x.reshape(-1,1))).item() for x in xs]
max_f = max(fs)
fs = [8*i/max_f for i in fs]
plt.plot(xs, fs, color="black", label=r"$\propto \exp{- (f(x) + g(x))}$")
plt.xlabel("x")
plt.ylabel("Number of samples")
plt.legend()
plt.show()

plt.plot(range(len(dist_ADMM)), dist_ADMM, label="ADMM sampler")
plt.plot(range(len(dist_cont_ADMM)), dist_cont_ADMM, label="ADMM cont")
# plt.plot(range(len(dist_PLA)), dist_PLA, label="PLA")
# plt.plot(range(len(dist_PD)), dist_PD, label="PD")
plt.xlabel("Iteration")
plt.ylabel("Wasserstein distance")
plt.legend()
plt.show()


# Mean Propagation
# r = params['rho']
# A = np.array([[0, r/(r+1), -r/(r+1), -r/(r+1)], [0, r/(r+1), 1/(r+1), -r/(r+1)], [0, 0, 0, 0], [0, 0, 0, 1]])
# mus = np.array([3, 3, 3, 0]).reshape(-1, 1)
# mu_seq = []
# mu_seq.append(mus[0, 0])
# for _ in range(len(dist_ADMM)-1):
#     mus = A@mus
#     mu_seq.append(mus[0, 0])
# plt.plot(range(len(dist_ADMM)), dist_ADMM, label="ADMM - empir")
# plt.plot(range(len(mu_seq)), mu_seq, label="ADMM - theor")

# gamma = params['gamma']
# mu = 3.
# mu_seq = []
# mu_seq.append(mu)
# for _ in range(len(dist_ADMM)-1):
#     mu = 1/(1+gamma)*mu
#     mu_seq.append(mu)
# plt.plot(range(len(dist_PLA)), dist_PLA, label="PLA - empir")
# plt.plot(range(len(mu_seq)), mu_seq, label="PLA - theor")



