import numpy as np
import scipy.special as sc
from numpy.linalg import inv, det

class DpGaussian2D:
    """
    2 Dimensional, Diagonal variance matrix, with equal variance in each dimension.
    """

    def __init__(self, x, alpha, sigma: float, mu0, sigma0: float, K) -> None:
        """
        alpha: parameter of DP
        sigma: variance of gaussian
        mu0: mean of the prior to mean of gaussian
        sigma0: variance of the prior to mean of gaussian
        """
        self.x = x
        self.alpha = alpha
        self.sigma = sigma
        self.lmbda1 = sigma ** 2 / sigma0 ** 2 * mu0
        self.lmbda2 = sigma ** 2 / sigma0 ** 2
        self.mu0 = mu0
        self.sigma0 = sigma0

        self.N = len(x)
        self.K = K

        self.theta = np.random.random((self.N, self.K))
        for theta_i in self.theta:
            theta_i /= np.sum(theta_i)

        self.gamma1 = np.random.random((self.K,))
        self.gamma2 = np.random.random((self.K,))
        self.tau1 = np.random.random((self.K, 2))
        self.tau2 = np.random.random((self.K,))

    def update(self):
        for i in range(self.K):
            self.gamma1[i] = 1 + np.sum(self.theta[:, i])
            self.gamma2[i] = self.alpha + np.sum(self.theta[:, i+1: self.K])
            self.tau1[i] = self.lmbda1 + np.dot(self.theta[:, i], self.x)
            self.tau2[i] = self.lmbda2 + np.sum(self.theta[:, i])

        E_mu_T_mu = np.sum(self.tau1 * self.tau1, axis=1) / self.tau2 ** 2 + 2 * self.sigma ** 2 / self.tau2

        for n in range(self.N):
            E = np.zeros(self.K)
            for i in range(self.K):
                E[i] = np.exp(sc.digamma(self.gamma1[i]) - sc.digamma(self.gamma1[i] + self.gamma2[i]) \
                     + np.dot(self.tau1[i], self.x[n]) / self.sigma ** 2 / self.tau2[i] \
                     - (E_mu_T_mu[i]) / 2 / self.sigma ** 2 \
                     + np.sum(sc.digamma(self.gamma2[0: i-1]) - sc.digamma(self.gamma1[0: i-1] + self.gamma2[0: i-1])))
            self.theta[n] = E / np.sum(E)
    
    def elbo(self) -> float:
        E_log_v = sc.digamma(self.gamma1) - sc.digamma(self.gamma1 + self.gamma2)
        E_log_neg_v = sc.digamma(self.gamma2) - sc.digamma(self.gamma1 + self.gamma2)
        E_mu_T_mu = np.sum(self.tau1 * self.tau1, axis=1) / self.tau2 ** 2 + 2 * self.sigma ** 2 / self.tau2

        E_log_p_V = - self.K * sc.beta(1, self.alpha) + (self.alpha - 1) * np.sum(E_log_neg_v)
        E_log_p_eta = - self.K * np.log(2 * np.pi * self.sigma0 ** 2)
        E_log_p_Z = 0
        E_log_p_x = - self.N * np.log(2 * np.pi * self.sigma ** 2)
        E_log_q_v = 0
        E_log_q_eta = - self.K * np.log(2 * np.pi * self.sigma ** 2)
        E_log_q_z = 0

        for i in range(self.K):
            E_log_p_eta += - (E_mu_T_mu[i] - 2 * np.dot(self.tau1[i], self.mu0) / self.tau2[i] + np.dot(self.mu0, self.mu0)) / 2 / self.sigma0 ** 2
            
            for n in range(self.N):
                E_log_p_Z += np.sum(self.theta[n, i+1: self.K]) * E_log_neg_v[i] + self.theta[n, i] * E_log_v[i] # speed up?
                E_log_p_x += - self.theta[n, i] * (np.dot(self.x[n], self.x[n]) - 2 * np.dot(self.tau1[i], self.x[n]) / self.tau2[i] + E_mu_T_mu[i]) / 2 / self.sigma ** 2
                E_log_q_z += self.theta[n, i] * np.log(self.theta[n, i])

            if i < self.K - 1:
                E_log_q_v += - np.log(sc.beta(self.gamma1[i], self.gamma2[i])) + (self.gamma1[i] - 1) * E_log_v[i] + (self.gamma2[i] - 1) * E_log_neg_v[i]

            E_log_q_eta += np.log(self.tau2[i]) - 1
        
        return E_log_p_V + E_log_p_eta + E_log_p_Z + E_log_p_x - E_log_q_v - E_log_q_eta - E_log_q_z

class DpGaussian:
    """
    A general DP Gaussian-Gaussian model.
    """

    def __init__(self, x, alpha, Lmbda, mu0, Lmbda0, K) -> None:
        """
        alpha: parameter of DP
        Lmbda: covariance matrix of gaussian
        mu0: mean of the prior to mean of gaussian
        Lmbda0: covariance matrix of the prior
        """
        self.x = x
        self.alpha = alpha
        self.Lmbda = Lmbda
        self.Lmbda0 = Lmbda0
        self.Lmbda_inv = inv(self.Lmbda)
        self.Lmbda0_inv = inv(self.Lmbda0)
        self.mu0 = mu0
        self.lmbda1 = self.Lmbda @ self.Lmbda0_inv @ self.mu0
        self.lmbda2 = 0

        self.N = np.size(x, 0)
        self.D = np.size(x, 1)
        self.K = K

        self.theta = np.random.random((self.N, self.K))
        for theta_i in self.theta:
            theta_i /= np.sum(theta_i)

        self.gamma1 = np.random.random((self.K,))
        self.gamma2 = np.random.random((self.K,))
        self.tau1 = np.random.random((self.K, 2))
        self.tau2 = np.random.random((self.K,))

    def update(self):
        for i in range(self.K):
            self.gamma1[i] = 1 + np.sum(self.theta[:, i])
            self.gamma2[i] = self.alpha + np.sum(self.theta[:, i+1: self.K])
            self.tau1[i] = self.lmbda1 + np.dot(self.theta[:, i], self.x)
            self.tau2[i] = self.lmbda2 + np.sum(self.theta[:, i])
            
        E_log_v = sc.digamma(self.gamma1) - sc.digamma(self.gamma1 + self.gamma2)
        E_log_neg_v = sc.digamma(self.gamma2) - sc.digamma(self.gamma1 + self.gamma2)

        Lmbda_p_inv = np.zeros((self.K, self.D, self.D))
        Lmbda_p = np.zeros((self.K, self.D, self.D))
        mu_p = np.zeros((self.K, self.D))
        E_a_eta = np.zeros((self.K,))
        for i in range(self.K):
            Lmbda_p_inv[i] = self.tau2[i] * self.Lmbda_inv + self.Lmbda0_inv
            Lmbda_p[i] = inv(Lmbda_p_inv[i])
            mu_p[i] = Lmbda_p[i] @ self.Lmbda_inv @ self.tau1[i]
            E_a_eta[i] = 0.5 * (mu_p[i] @ self.Lmbda_inv @ mu_p[i] + np.sum(self.Lmbda_inv * Lmbda_p[i]))

        for n in range(self.N):
            E = np.zeros(self.K)
            for i in range(self.K):
                E[i] = np.exp(E_log_v[i] + mu_p[i] @ self.Lmbda_inv @ self.x[n] - E_a_eta[i] + np.sum(E_log_neg_v[0: i - 1]))
            self.theta[n] = E / np.sum(E)

    def elbo(self) -> float:
        E_log_v = sc.digamma(self.gamma1) - sc.digamma(self.gamma1 + self.gamma2)
        E_log_neg_v = sc.digamma(self.gamma2) - sc.digamma(self.gamma1 + self.gamma2)

        E_log_p_V = - self.K * sc.beta(1, self.alpha) + (self.alpha - 1) * np.sum(E_log_neg_v)
        E_log_p_eta = - self.K * (self.D / 2 * np.log(2 * np.pi) + 0.5 * np.log(det(self.Lmbda0)))
        E_log_p_Z = 0
        E_log_p_x = - self.N * (self.D / 2 * np.log(2 * np.pi) + 0.5 * np.log(det(self.Lmbda0)))
        E_log_q_v = 0
        E_log_q_eta = - self.K * (self.D / 2 * np.log(2 * np.pi) - 0.5 * self.D ** 2)
        E_log_q_z = 0

        Lmbda_p_inv = np.zeros((self.K, self.D, self.D))
        Lmbda_p = np.zeros((self.K, self.D, self.D))
        mu_p = np.zeros((self.K, self.D))
        for i in range(self.K):
            Lmbda_p_inv[i] = self.tau2[i] * self.Lmbda_inv + self.Lmbda0_inv
            Lmbda_p[i] = inv(Lmbda_p_inv[i])
            mu_p[i] = Lmbda_p[i] @ self.Lmbda_inv @ self.tau1[i]

        for i in range(self.K):
            E_log_p_eta += - 0.5 * (mu_p[i] @ self.Lmbda0_inv @ mu_p[i] + np.sum(self.Lmbda0_inv * Lmbda_p[i]) - 2 * self.mu0 @ self.Lmbda0_inv @ mu_p[i] + self.mu0 @ self.Lmbda0_inv @ self.mu0)
            
            for n in range(self.N):
                E_log_p_Z += np.sum(self.theta[n, i+1: self.K]) * E_log_neg_v[i] + self.theta[n, i] * E_log_v[i] # speed up?
                E_log_p_x += - self.theta[n, i] * 0.5 * (self.x[n] @ self.Lmbda_inv @ self.x[n] - 2 * self.x[n] @ self.Lmbda_inv @ mu_p[i] + mu_p[i] @ self.Lmbda_inv @ mu_p[i] + np.sum(self.Lmbda_inv * Lmbda_p[i]))
                E_log_q_z += self.theta[n, i] * np.log(self.theta[n, i])

            if i < self.K - 1:
                E_log_q_v += - np.log(sc.beta(self.gamma1[i], self.gamma2[i])) + (self.gamma1[i] - 1) * E_log_v[i] + (self.gamma2[i] - 1) * E_log_neg_v[i]

            E_log_q_eta += - 0.5 * np.log(det(Lmbda_p[i]))
        
        return E_log_p_V + E_log_p_eta + E_log_p_Z + E_log_p_x - E_log_q_v - E_log_q_eta - E_log_q_z

