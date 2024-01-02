import numpy as np
import scipy.special as sc
from numpy.linalg import inv, det

log_2pi = np.log(2 * np.pi)
eps = 1e-9

class DpGaussianSpherical2D:
    """
    2 Dimensional, Spherical covariance matrix
    """

    def __init__(self, x, alpha, sigma: float, mu0, sigma0: float, K: int) -> None:
        """
        x: array with size (N, 2)
        alpha: parameter of DP
        sigma: variance of gaussian
        mu0: mean of the prior to mean of gaussian, size (2,)
        sigma0: variance of the prior to mean of gaussian
        K: truncation level
        """
        self.x = x
        self.alpha = alpha
        self.sigma = sigma
        self.lmbda1 = sigma ** 2 / sigma0 ** 2 * mu0
        self.lmbda2 = sigma ** 2 / sigma0 ** 2
        self.mu0 = mu0
        self.sigma0 = sigma0

        self.N = np.size(x, 0)
        self.D = np.size(x, 1)
        self.K = K

        self.phi = np.random.random((self.N, self.K))
        for phi_i in self.phi:
            phi_i /= np.sum(phi_i)

        self.gamma1 = np.random.random((self.K,))
        self.gamma2 = np.random.random((self.K,))
        self.tau1 = np.random.random((self.K, 2))
        self.tau2 = np.random.random((self.K,))
        
        self.mu_p = np.zeros((self.K, self.D))
        self.sigma_p = np.zeros((self.K,))
        for i in range(self.K):
            self.mu_p[i] = self.tau1[i] / self.tau2[i]
            self.sigma_p[i] = self.sigma ** 2 / self.tau2[i]

    def update(self):
        """
        A single step of CAVI
        """
        for i in range(self.K):
            self.gamma1[i] = 1 + np.sum(self.phi[:, i])
            self.gamma2[i] = self.alpha + np.sum(self.phi[:, i+1: self.K])
            self.tau1[i] = self.lmbda1 + np.dot(self.phi[:, i], self.x)
            self.tau2[i] = self.lmbda2 + np.sum(self.phi[:, i])

        E_mu_T_mu = np.sum(self.tau1 * self.tau1, axis=1) / self.tau2 ** 2 + 2 * self.sigma ** 2 / self.tau2
        E_log_v = sc.digamma(self.gamma1) - sc.digamma(self.gamma1 + self.gamma2)
        E_log_v[-1] = 0
        E_log_neg_v = sc.digamma(self.gamma2) - sc.digamma(self.gamma1 + self.gamma2)

        for n in range(self.N):
            E = np.zeros(self.K)
            for i in range(self.K):
                E[i] = np.exp(E_log_v[i] \
                     + np.dot(self.tau1[i], self.x[n]) / self.sigma ** 2 / self.tau2[i] \
                     - (E_mu_T_mu[i]) / 2 / self.sigma ** 2 \
                     + np.sum(E_log_neg_v[0: i]))
            self.phi[n] = E / np.sum(E)

        for i in range(self.K):
            self.mu_p[i] = self.tau1[i] / self.tau2[i]
            self.sigma_p[i] = self.sigma ** 2 / self.tau2[i]
    
    def elbo(self) -> float:
        """
        Return current elbo
        """
        E_log_v = sc.digamma(self.gamma1) - sc.digamma(self.gamma1 + self.gamma2)
        E_log_neg_v = sc.digamma(self.gamma2) - sc.digamma(self.gamma1 + self.gamma2)
        E_mu_T_mu = np.sum(self.mu_p * self.mu_p, axis=1) + 2 * self.sigma ** 2 / self.tau2

        E_log_p_V = - self.K * sc.beta(1, self.alpha) + (self.alpha - 1) * np.sum(E_log_neg_v)
        E_log_p_eta = - self.K * np.log(2 * np.pi * self.sigma0 ** 2)
        E_log_p_Z = 0
        E_log_p_x = - self.N * np.log(2 * np.pi * self.sigma ** 2)
        E_log_q_v = 0
        E_log_q_eta = - self.K * np.log(2 * np.pi * self.sigma ** 2)
        E_log_q_z = 0

        for i in range(self.K):
            E_log_p_eta += - (E_mu_T_mu[i] - 2 * np.dot(self.mu_p[i], self.mu0) + np.dot(self.mu0, self.mu0)) / 2 / self.sigma0 ** 2
            
            for n in range(self.N):
                E_log_p_Z += np.sum(self.phi[n, i+1: self.K]) * E_log_neg_v[i] + self.phi[n, i] * E_log_v[i] # speed up?
                E_log_p_x += - self.phi[n, i] * (np.dot(self.x[n], self.x[n]) - 2 * np.dot(self.tau1[i], self.x[n]) / self.tau2[i] + E_mu_T_mu[i]) / 2 / self.sigma ** 2
                E_log_q_z += self.phi[n, i] * np.log(self.phi[n, i])

            if i < self.K - 1:
                E_log_q_v += - sc.betaln(self.gamma1[i], self.gamma2[i]) + (self.gamma1[i] - 1) * E_log_v[i] + (self.gamma2[i] - 1) * E_log_neg_v[i]

            E_log_q_eta += np.log(self.tau2[i]) - 1
        
        return E_log_p_V + E_log_p_eta + E_log_p_Z + E_log_p_x - E_log_q_v - E_log_q_eta - E_log_q_z
    
    def predict(self, x):
        """
        Input:
            x: an array of dataset, with size (M, D)
        Output: 
            An array of size (M, K):
                out[n, i] is the likelihood of "the model generates x_n, and it is in i-th component." 

                np.sum(out, axis=1) gives the likelihood of x_n
                
                np.argmax(out, axis=1) gives the label of x_n
        """
        # posterior variance
        var_x = self.sigma ** 2 + self.sigma_p ** 2
        # prior probability of component i
        log_p_k = np.zeros((self.K,))
        temp_sum = 0
        for i in range(self.K):
            if i < self.K - 1:
                log_p_k[i] = temp_sum + np.log(self.gamma1[i]) - np.log(self.gamma1[i] + self.gamma2[i])
            else:
                log_p_k[i] = temp_sum 
            temp_sum += np.log(self.gamma2[i]) - np.log(self.gamma1[i] + self.gamma2[i])

        M = np.size(x, 0)
        log_p_x_k = np.ones((M, self.K)) * (- log_2pi)
        for n in range(M):
            for i in range(self.K):
                # probability of x_n, if in component i
                log_p_x_k[n, i] += - np.log(var_x[i]) - 0.5 * np.dot(x[n] - self.mu_p[i], x[n] - self.mu_p[i]) / var_x[i]

        return log_p_k + log_p_x_k

class DpGaussianSpherical:
    """
    Spherical covariance matrix.
    """

    def __init__(self, x, alpha, sigma: float, mu0, sigma0: float, K: int) -> None:
        """
        x: array with size (N, D)
        alpha: parameter of DP
        sigma: variance of gaussian
        mu0: mean of the prior to mean of gaussian, size (D,)
        sigma0: variance of the prior to mean of gaussian
        K: truncation level
        """
        self.x = x
        self.alpha = alpha
        self.sigma = sigma
        self.lmbda1 = sigma ** 2 / sigma0 ** 2 * mu0
        self.lmbda2 = sigma ** 2 / sigma0 ** 2
        self.mu0 = mu0
        self.sigma0 = sigma0

        self.N = np.size(x, 0)
        self.D = np.size(x, 1)
        self.K = K

        self.phi = np.random.random((self.N, self.K))
        for phi_i in self.phi:
            phi_i /= np.sum(phi_i)

        self.gamma1 = np.random.random((self.K,))
        self.gamma2 = np.random.random((self.K,))
        self.tau1 = np.random.random((self.K, self.D))
        self.tau2 = np.random.random((self.K,))
        
        self.mu_p = np.zeros((self.K, self.D))
        self.sigma_p = np.zeros((self.K,))
        for i in range(self.K):
            self.mu_p[i] = self.tau1[i] / self.tau2[i]
            self.sigma_p[i] = self.sigma ** 2 / self.tau2[i]

    def update(self):
        """
        A single step of CAVI
        """
        for i in range(self.K):
            self.gamma1[i] = 1 + np.sum(self.phi[:, i])
            self.gamma2[i] = self.alpha + np.sum(self.phi[:, i+1: self.K])
            self.tau1[i] = self.lmbda1 + np.dot(self.phi[:, i], self.x)
            self.tau2[i] = self.lmbda2 + np.sum(self.phi[:, i])

        E_mu_T_mu = np.sum(self.tau1 * self.tau1, axis=1) / self.tau2 ** 2 + self.D * self.sigma ** 2 / self.tau2
        E_log_v = sc.digamma(self.gamma1) - sc.digamma(self.gamma1 + self.gamma2)
        E_log_v[-1] = 0
        E_log_neg_v = sc.digamma(self.gamma2) - sc.digamma(self.gamma1 + self.gamma2)

        for n in range(self.N):
            E = np.zeros(self.K)
            for i in range(self.K):
                E[i] = np.exp(E_log_v[i] \
                     + np.dot(self.tau1[i], self.x[n]) / self.sigma ** 2 / self.tau2[i] \
                     - (E_mu_T_mu[i]) / 2 / self.sigma ** 2 \
                     + np.sum(E_log_neg_v[0: i]))
            self.phi[n] = E / np.sum(E)

        for i in range(self.K):
            self.mu_p[i] = self.tau1[i] / self.tau2[i]
            self.sigma_p[i] = self.sigma ** 2 / self.tau2[i]
    
    def elbo(self) -> float:
        """
        Return current elbo
        """
        E_log_v = sc.digamma(self.gamma1) - sc.digamma(self.gamma1 + self.gamma2)
        E_log_neg_v = sc.digamma(self.gamma2) - sc.digamma(self.gamma1 + self.gamma2)
        E_mu_T_mu = np.sum(self.mu_p * self.mu_p, axis=1) + self.D * self.sigma ** 2 / self.tau2

        E_log_p_V = - self.K * sc.beta(1, self.alpha) + (self.alpha - 1) * np.sum(E_log_neg_v)
        E_log_p_eta = - self.K * self.D * (0.5 * log_2pi + np.log(self.sigma0))
        E_log_p_Z = 0
        E_log_p_x = - self.N * self.D * (0.5 * log_2pi + np.log(self.sigma))
        E_log_q_v = 0
        E_log_q_eta = - self.K * self.D * (0.5 * log_2pi + np.log(self.sigma))
        E_log_q_z = 0

        for i in range(self.K):
            E_log_p_eta += - (E_mu_T_mu[i] - 2 * np.dot(self.mu_p[i], self.mu0) + np.dot(self.mu0, self.mu0)) / 2 / self.sigma0 ** 2
            
            for n in range(self.N):
                E_log_p_Z += np.sum(self.phi[n, i+1: self.K]) * E_log_neg_v[i] + self.phi[n, i] * E_log_v[i] # speed up?
                E_log_p_x += - self.phi[n, i] * (np.dot(self.x[n], self.x[n]) - 2 * np.dot(self.tau1[i], self.x[n]) / self.tau2[i] + E_mu_T_mu[i]) / 2 / self.sigma ** 2
                E_log_q_z += self.phi[n, i] * np.log(self.phi[n, i])

            if i < self.K - 1:
                E_log_q_v += - sc.betaln(self.gamma1[i], self.gamma2[i]) + (self.gamma1[i] - 1) * E_log_v[i] + (self.gamma2[i] - 1) * E_log_neg_v[i]

            E_log_q_eta += np.log(self.tau2[i]) - 1
        
        return E_log_p_V + E_log_p_eta + E_log_p_Z + E_log_p_x - E_log_q_v - E_log_q_eta - E_log_q_z
    
    def predict(self, x):
        """
        Input:
            x: an array of dataset, with size (M, D)
        Output: 
            An array of size (M, K):
                out[n, i] is the log likelihood of "the model generates x_n, and it is in i-th component." 

                logsumexp(out, axis=1) gives the log likelihood of x_n
                
                np.argmax(out, axis=1) gives the label of x_n
        """
        # posterior variance
        var_x = self.sigma ** 2 + self.sigma_p ** 2
        # prior probability of component i
        log_p_k = np.zeros((self.K,))
        temp_sum = 0
        for i in range(self.K):
            if i < self.K - 1:
                log_p_k[i] = temp_sum + np.log(self.gamma1[i]) - np.log(self.gamma1[i] + self.gamma2[i])
            else:
                log_p_k[i] = temp_sum 
            temp_sum += np.log(self.gamma2[i]) - np.log(self.gamma1[i] + self.gamma2[i])

        M = np.size(x, 0)
        log_p_x_k = np.ones((M, self.K)) * (- 0.5 * self.D * log_2pi)
        for n in range(M):
            for i in range(self.K):
                # probability of x_n, if in component i
                log_p_x_k[n, i] += - np.log(var_x[i]) - 0.5 * np.dot(x[n] - self.mu_p[i], x[n] - self.mu_p[i]) / var_x[i]

        return log_p_k + log_p_x_k

class DpGaussianFull:
    """
    A general DP Gaussian-Gaussian model. Full covariance matrix
    """

    def __init__(self, x, alpha: float, Sigma, mu0, Sigma0, K: int) -> None:
        """
        x: array with size (N, D)
        alpha: parameter of DP
        Sigma: covariance matrix of gaussian
        mu0: mean of the prior to mean of gaussian
        Sigma0: covariance matrix of the prior
        K: truncation level
        """
        self.x = x
        self.alpha = alpha
        self.Sigma = Sigma
        self.Sigma0 = Sigma0
        self.Sigma_inv = inv(self.Sigma)
        self.Sigma0_inv = inv(self.Sigma0)
        self.mu0 = mu0
        self.lmbda1 = self.Sigma @ self.Sigma0_inv @ self.mu0
        self.lmbda2 = 0

        self.N = np.size(x, 0)
        self.D = np.size(x, 1)
        self.K = K

        self.phi = np.random.random((self.N, self.K))
        for phi_i in self.phi:
            phi_i /= np.sum(phi_i)

        self.gamma1 = np.random.random((self.K,))
        self.gamma2 = np.random.random((self.K,))
        self.tau1 = np.random.random((self.K, self.D))
        self.tau2 = np.random.random((self.K,))

        self.Sigma_p = np.zeros((self.K, self.D, self.D))
        self.Sigma_p_inv = np.zeros((self.K, self.D, self.D))
        self.mu_p = np.zeros((self.K, self.D))

    def update(self):
        for i in range(self.K):
            self.gamma1[i] = 1 + np.sum(self.phi[:, i])
            self.gamma2[i] = self.alpha + np.sum(self.phi[:, i+1: self.K])
            self.tau1[i] = self.lmbda1 + np.dot(self.phi[:, i], self.x)
            self.tau2[i] = self.lmbda2 + np.sum(self.phi[:, i])
            
        E_log_v = sc.digamma(self.gamma1) - sc.digamma(self.gamma1 + self.gamma2)
        E_log_v[-1] = 0
        E_log_neg_v = sc.digamma(self.gamma2) - sc.digamma(self.gamma1 + self.gamma2)

        Sigma_p_inv = np.zeros((self.K, self.D, self.D))
        Sigma_p = np.zeros((self.K, self.D, self.D))
        mu_p = np.zeros((self.K, self.D))
        E_a_eta = np.zeros((self.K,))

        for i in range(self.K):
            Sigma_p_inv[i] = self.tau2[i] * self.Sigma_inv + self.Sigma0_inv
            Sigma_p[i] = inv(Sigma_p_inv[i])
            mu_p[i] = Sigma_p[i] @ self.Sigma_inv @ self.tau1[i]
            E_a_eta[i] = 0.5 * (mu_p[i] @ self.Sigma_inv @ mu_p[i] + np.sum(self.Sigma_inv * Sigma_p[i]))

        for n in range(self.N):
            E = np.zeros(self.K)
            for i in range(self.K):
                E[i] = np.exp(E_log_v[i] + mu_p[i] @ self.Sigma_inv @ self.x[n] - E_a_eta[i] + np.sum(E_log_neg_v[0: i]))
            self.phi[n] = E / np.sum(E)

        for i in range(self.K):
            self.Sigma_p_inv[i] = self.tau2[i] * self.Sigma_inv + self.Sigma0_inv
            self.Sigma_p[i] = inv(Sigma_p_inv[i])
            self.mu_p[i] = Sigma_p[i] @ self.Sigma_inv @ self.tau1[i]

    def elbo(self) -> float:
        E_log_v = sc.digamma(self.gamma1) - sc.digamma(self.gamma1 + self.gamma2)
        E_log_neg_v = sc.digamma(self.gamma2) - sc.digamma(self.gamma1 + self.gamma2)

        E_log_p_V = - self.K * sc.beta(1, self.alpha) + (self.alpha - 1) * np.sum(E_log_neg_v)
        E_log_p_eta = - self.K * (self.D / 2 * log_2pi + 0.5 * np.log(det(self.Sigma0)))
        E_log_p_Z = 0
        E_log_p_x = - self.N * (self.D / 2 * log_2pi + 0.5 * np.log(det(self.Sigma0)))
        E_log_q_v = 0
        E_log_q_eta = - self.K * (self.D / 2 * log_2pi - 0.5 * self.D ** 2)
        E_log_q_z = 0

        for i in range(self.K):
            E_log_p_eta += - 0.5 * (self.mu_p[i] @ self.Sigma0_inv @ self.mu_p[i] + np.sum(self.Sigma0_inv * self.Sigma_p[i]) - 2 * self.mu0 @ self.Sigma0_inv @ self.mu_p[i] + self.mu0 @ self.Sigma0_inv @ self.mu0)
            
            for n in range(self.N):
                E_log_p_Z += np.sum(self.phi[n, i+1: self.K]) * E_log_neg_v[i] + self.phi[n, i] * E_log_v[i] # speed up?
                E_log_p_x += - self.phi[n, i] * 0.5 * (self.x[n] @ self.Sigma_inv @ self.x[n] - 2 * self.x[n] @ self.Sigma_inv @ self.mu_p[i] + self.mu_p[i] @ self.Sigma_inv @ self.mu_p[i] + np.sum(self.Sigma_inv * self.Sigma_p[i]))
                E_log_q_z += self.phi[n, i] * np.log(self.phi[n, i] + eps)

            if i < self.K - 1:
                E_log_q_v += - sc.betaln(self.gamma1[i], self.gamma2[i]) + (self.gamma1[i] - 1) * E_log_v[i] + (self.gamma2[i] - 1) * E_log_neg_v[i]

            E_log_q_eta += - 0.5 * np.log(det(self.Sigma_p[i]))
        
        return E_log_p_V + E_log_p_eta + E_log_p_Z + E_log_p_x - E_log_q_v - E_log_q_eta - E_log_q_z
    
    def predict(self, x):
        """
        Input:
            x: an array of dataset, with size (M, D)
        Output: 
            An array of size (M, K):
                out[n, i] is the likelihood of "the model generates x_n, and it is in i-th component." 

                np.sum(out, axis=1) gives the likelihood of x_n
                
                np.argmax(out, axis=1) gives the label of x_n
        """
        # posterior covariance
        cov_x = self.Sigma + self.Sigma_p
        cov_x_inv = inv(cov_x)
        log_det_cov_x = np.log(det(cov_x))
        # prior probability of component i
        log_p_k = np.zeros((self.K,))
        temp_sum = 0
        for i in range(self.K):
            if i < self.K - 1:
                log_p_k[i] = temp_sum + np.log(self.gamma1[i]) - np.log(self.gamma1[i] + self.gamma2[i])
            else:
                log_p_k[i] = temp_sum 
            temp_sum += np.log(self.gamma2[i]) - np.log(self.gamma1[i] + self.gamma2[i])

        M = np.size(x, 0)
        log_p_x_k = np.ones((M, self.K)) * (- self.D / 2 * log_2pi)
        for n in range(M):
            for i in range(self.K):
                # probability of x_n, if in component i
                log_p_x_k[n, i] += - 0.5 * log_det_cov_x[i] - 0.5 * (x[n] - self.mu_p[i]) @ cov_x_inv[i] @ (x[n] - self.mu_p[i]) 

        return log_p_k + log_p_x_k