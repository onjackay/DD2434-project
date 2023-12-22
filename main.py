import numpy as np
import scipy.special as sc
import matplotlib.pyplot as plt

from dp import *

# np.random.seed(0)
N = 100
K = 20
x1 = np.random.normal(loc=-10, scale=1, size=(50, 2))
x2 = np.random.normal(loc=10, scale=1, size=(50, 2))
x = np.concatenate((x1, x2), axis=0)

plt.figure()
plt.scatter(x[:, 0], x[:, 1])
plt.show()

alpha = 1
sigma = 1
mu0 = np.zeros((2,))
sigma0 = 1
dp = DpGaussian2D(x, alpha, sigma, mu0, sigma0, K)
for i in range(20):
    dp.update()
    print(dp.elbo())

# for n in range(N):
    # print(dp.theta[n])

print(np.sum(dp.theta, axis=0))

# for i in range(dp.K):
#     print(dp.sigma ** 2 / dp.tau2[i])
#     print(dp.tau1[i] / dp.tau2[i])
#     print()

# for i in range(K):
#     print(dp.gamma1[i], dp.gamma2[i])

# Lmbda = np.diag(np.ones((2,)))
# Lmbda0 = np.diag(np.ones((2,)))
# dp2 = DpGaussian(x, alpha, Lmbda, mu0, Lmbda0, K)
# for i in range(20):
#     dp2.update()
    # print(dp2.elbo())

# Lmbda_p_inv = np.zeros((dp2.K, dp2.D, dp2.D))
# Lmbda_p = np.zeros((dp2.K, dp2.D, dp2.D))
# mu_p = np.zeros((dp2.K, dp2.D))
# for i in range(dp2.K):            
#     Lmbda_p_inv[i] = dp2.tau2[i] * dp2.Lmbda_inv + dp2.Lmbda0_inv
#     Lmbda_p[i] = inv(Lmbda_p_inv[i])
#     mu_p[i] = Lmbda_p[i] @ dp2.Lmbda_inv @ dp2.tau1[i]
#     print(Lmbda_p[i])
#     print(mu_p[i])
#     print()