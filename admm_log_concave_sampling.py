###########################################################################
# Import packages
import numpy as np
import dit
import numpy.random as rnd
from pyhmc import hmc
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial import distance_matrix
import scipy as sp
from tqdm import tqdm

###########################################################################
# Algorithm Parameters
params = {}
params["num_samples"] = 500
params["num_iters"] = 15
params["dim_x"] = 1
params["dim_z"] = 1
params["rho"] = 2.5
params["h"] = 0.1
params["gamma"] = 0.5
params["γ_KLMC"] = 100
params["h_KLMC"] = 0.0001
params["η_HMC"] = 0.1
params["γ_HMC"] = 7.

# Gaussian objective
target_mean = 4*rnd.normal(size=(params["dim_x"], ))+2*rnd.uniform(size=(params["dim_x"],))
# target_mean = np.array([0.])
params["target_mean"] = target_mean

target_covar = 0.1*rnd.normal(size=(params["dim_x"], params["dim_x"]))
target_covar = target_covar.T@target_covar + 0.1*np.identity(params["dim_x"])
# target_covar = np.array([3]).reshape(1,1)
params["target_covar"] = target_covar

# Samples from target
params["target_samples"] = rnd.multivariate_normal(mean=target_mean, cov=target_covar, size=(params["num_samples"]))

# Gradient of target distribution
grad_f = lambda x: np.linalg.inv(target_covar)@(x-target_mean)
norm_target_density = lambda x: np.exp(-0.5*np.transpose(x-target_mean)@np.linalg.inv(target_covar)@(x-target_mean))

###########################################################################
# Helper functions
def wass_dist(xs, target_mean, target_covar):
    empirical_mean = np.mean(xs.T, axis=0)
    empirical_cov  = EmpiricalCovariance(assume_centered=False).fit(xs.T).covariance_

    dist = np.linalg.norm(empirical_mean - target_mean)**2 + \
            np.trace(empirical_cov + target_covar - 2*sp.linalg.sqrtm( sp.linalg.sqrtm(target_covar)@empirical_cov@sp.linalg.sqrtm(target_covar)  ) )
    
    dist = np.sqrt(dist)

    return(dist)

# def logprob(x, target_mean, target_covar):
#     _x = x.reshape(-1)
#     logp = -0.5*np.transpose(_x-target_mean)@np.linalg.inv(target_covar)@(_x-target_mean)
#     grad_logp = np.linalg.inv(target_covar)@(_x-target_mean)

#     return logp, grad_logp

# def emd2dist(source_samples, target_samples):
#     M = np.square(distance_matrix(source_samples.T, target_samples))
    
#     dist_squared = dit.divergences.earth_movers_distance(1/source_samples.shape[1]*np.ones((source_samples.shape[1], 1)), \
#                                              1/target_samples.shape[0]*np.ones((target_samples.shape[0], 1)), M)
#     return np.sqrt(dist_squared)


###########################################################################
def run_methods(params):

    num_samples = params["num_samples"]
    num_iters = params["num_iters"]

    h = params["h"]
    rho = params["rho"]
    gamma = params["gamma"]

    γ_KLMC = params["γ_KLMC"]
    h_KLMC = params["h_KLMC"]
    ψ0 = np.exp(-γ_KLMC*h_KLMC)
    ψ1 = 1/γ_KLMC*(1-ψ0) 
    ψ2 = (γ_KLMC*h_KLMC + np.exp(-γ_KLMC*h_KLMC) - 1)/(γ_KLMC**2)

    η_HMC = params["η_HMC"]
    γ_HMC = params["γ_HMC"]

    dim_x = params["dim_x"]
    dim_z = params["dim_z"]

    target_mean = params["target_mean"]
    target_covar = params["target_covar"]
    target_samples = params["target_samples"]

    METHOD = params["METHOD"]

    # Set-up samples and output
    x = rnd.normal(loc=0.2, size=(dim_x, num_samples))
    z = rnd.normal(size=(dim_x, num_samples)) 
    p = np.zeros((dim_x, num_samples))
    v = np.zeros((dim_x, num_samples))
    wasserstein_dists = []
    emd_dists = []
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
            # emd_dists.append(emd2dist(x, target_samples=target_samples))
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

    elif METHOD == "MH":
        for iter in tqdm(range(num_iters)):
            wasserstein_dists.append(wass_dist(x, target_mean, target_covar))
            for sample in range(num_samples):
                prev_sample = x[:, sample]
                cand_sample = rnd.multivariate_normal(prev_sample, np.eye(dim_x)).reshape(-1)

                Q_given_cand = sp.stats.multivariate_normal(cand_sample)
                Q_given_prev = sp.stats.multivariate_normal(prev_sample)
                accept_prob = norm_target_density(cand_sample)/norm_target_density(prev_sample)*Q_given_cand.pdf(prev_sample)/Q_given_prev.pdf(cand_sample)
                if np.random.uniform(0, 1) < accept_prob:
                    x[:, sample] = cand_sample

    elif METHOD == "KLMC":
        covariance_noise = np.zeros((2*dim_x, 2*dim_x))
        for i in range(dim_x):
            covariance_noise[i,i] = (1-np.exp(-2*γ_KLMC*h_KLMC))/(2*γ_KLMC)
            covariance_noise[i+dim_x, i] = np.exp(-2*γ_KLMC*h_KLMC)*(np.exp(γ_KLMC*h_KLMC)-1)**2/(2*γ_KLMC**2)
            covariance_noise[i, i+dim_x] = np.exp(-2*γ_KLMC*h_KLMC)*(np.exp(γ_KLMC*h_KLMC)-1)**2/(2*γ_KLMC**2)
            covariance_noise[i+dim_x, i+dim_x] = -(-2*γ_KLMC*h_KLMC + np.exp(-2*γ_KLMC*h_KLMC) - 4*np.exp(-γ_KLMC*h_KLMC)+3)/(2*γ_KLMC**3)

        for iter in tqdm(range(num_iters)):
            wasserstein_dists.append(wass_dist(x, target_mean, target_covar))
            for sample in range(num_samples):
                noise_sample = rnd.multivariate_normal(np.zeros(2*dim_x), covariance_noise)
                theta_k = x[:, sample]
                v_k = v[:, sample]
                v_next_k = ψ0*v_k - ψ1*grad_f(theta_k) + np.sqrt(2*γ_KLMC)*noise_sample[:dim_x]
                theta_next_k = theta_k + ψ1*v_k - ψ2*grad_f(theta_k) + np.sqrt(2*γ_KLMC)*noise_sample[dim_x:]
                x[:, sample] = theta_next_k
                v[:, sample] = v_next_k

    elif METHOD == "HMC":
        for iter in tqdm(range(num_iters)):
            wasserstein_dists.append(wass_dist(x, target_mean, target_covar))
            for sample in range(num_samples):
                theta_k = x[:, sample]
                v_k = v[:, sample]
                v_next_k = v_k - η_HMC*(γ_HMC*v_k + grad_f(theta_k)) + np.sqrt(2*γ_HMC*η_HMC)*rnd.normal(size=(dim_x,))
                theta_next_k = theta_k + η_HMC*v_next_k
                x[:, sample] = theta_next_k
                v[:, sample] = v_next_k

    elif METHOD == "PLA_1":
        x_var = cp.Variable(dim_x)
        y_par = cp.Parameter(dim_x)

        x_objective = cp.Minimize( 0.5*cp.quad_form(x_var-target_mean, np.linalg.inv(target_covar))\
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

    elif METHOD == "PLA_2":
        x_var = cp.Variable(dim_x)
        y_par = cp.Parameter(dim_x)

        x_objective = cp.Minimize(   0.25*cp.quad_form(x_var-target_mean, np.linalg.inv(target_covar))\
                                    + 1/(2*gamma)*cp.norm( x_var - y_par)**2)
        x_problem = cp.Problem(x_objective)
        
        for iter in tqdm(range(num_iters)):
            wasserstein_dists.append(wass_dist(x, target_mean, target_covar))
            # means.append(np.mean(x.T, axis=0).item())
            # covars.append(EmpiricalCovariance(assume_centered=False).fit(x.T).covariance_.item())
            for sample in range(num_samples):
                y_par.value = x[:, sample] + np.sqrt(2*gamma)*rnd.normal(size=(dim_x,))
                
                x_problem.solve()
                y_par.value = x_var.value

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
# params["ADMM type"] = "g=0"
# # # # # dist_cont_ADMM, x_cont_ADMM = cont_time_ADMM(params)
# dist_ADMM_0, x_ADMM_0 = run_methods(params)
params["ADMM type"] = "f=g"
# # dist_cont_ADMM, x_cont_ADMM = cont_time_ADMM(params)
dist_ADMM_fg, x_ADMM_fg = run_methods(params)

# params["METHOD"] = "LGD"
# dist_LGD, x_LGD = run_methods(params)

params["METHOD"] = "PLA_1"
dist_PLA_1, x_PLA_1 = run_methods(params)

# params["METHOD"] = "PLA_2"
# dist_PLA_2, x_PLA_2 = run_methods(params)

# params["METHOD"] = "MH"
# dist_MH, x_MH = run_methods(params)

# params["METHOD"] == "KLMC"
# dist_KLMC, x_KLMC = run_methods(params)

# params["METHOD"] == "HMC"
# dist_HMC, x_HMC = run_methods(params)


# Plot results
plt.hist(x_ADMM_fg[0, :], alpha=0.4, bins=100, label="ADMM sampler")
# plt.hist(x_cont_ADMM[0, :], alpha=0.4, bins=100, label="ADMM cont")
plt.hist(x_PLA_1[0, :], alpha=0.4, bins=100, color='red', label="PLA")
# plt.hist(x_PD[0, :], alpha=0.4, bins=100, color='green', label="PD")

f = lambda x: 0.5*(x-target_mean).T@np.linalg.inv(target_covar)@(x-target_mean)
# f = lambda x: np.abs(x)
xs = list(np.arange(-5, 5., 0.02))
fs = [np.exp(-f(x.reshape(-1,1))).item() for x in xs]
max_f = max(fs)
fs = [25*i/max_f for i in fs]
plt.plot(xs, fs, color="black", label=r"$\propto \exp{- (f(x) + g(x))}$")
plt.xlabel("x")
plt.ylabel("Number of samples")
plt.legend()
plt.show()

# plt.plot(range(len(dist_ADMM_0)), dist_ADMM_0, label="ADMM (g=0) (ρ="+str(params["rho"])+")")
# plt.plot(range(len(dist_ADMM_fg)), dist_ADMM_fg, label="ADMM (f=g) analytic (ρ="+str(params["rho"])+")")
# plt.plot(range(len(dist_PLA_1)), dist_PLA_1, label="PLA unique g (γ="+str(params["gamma"])+")")
# plt.plot(range(len(dist_PLA_2)), dist_PLA_2, label="PLA g1=g2 (γ="+str(params["gamma"])+")")
# plt.plot(range(len(dist_LGD)), dist_LGD, label="LGD (h="+str(params["h"])+")")
# plt.plot(range(len(dist_MH)), dist_MH, label="MH")
# plt.plot(range(len(dist_KLMC)), dist_KLMC, label="KLMC")
# plt.plot(range(len(dist_HMC)), dist_HMC, label="HMC")
# plt.xlabel("Iteration")
# plt.ylabel("Wasserstein distance")
# plt.title("d="+str(params["dim_x"]))
# plt.legend()
# plt.show()


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




