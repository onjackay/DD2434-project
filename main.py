import numpy as np
import scipy.special as sc
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from scipy.io import arff
from numpy.lib.recfunctions import structured_to_unstructured
from timeit import default_timer

from dp import *

data, meta = arff.loadarff('dataset/dataset_2175_kin8nm.arff')
data = structured_to_unstructured(data)
N = 7000
K = 20
train_data = data[:N, :-1]

bgm = BayesianGaussianMixture(n_components=K, weight_concentration_prior=1e-5, n_init=1, covariance_type="diag", max_iter=1000, verbose=2)
bgm.fit(train_data)
print(bgm.n_features_in_)
print(bgm.weights_)
print(bgm.means_)

print()

print(bgm.covariances_)

print()

# cov_X = np.cov(train_data.T)
# mu_X = np.mean(train_data, axis=0)

# dp = DpGaussian(x=train_data, alpha=1/K, Lmbda=cov_X, mu0=mu_X, Lmbda0=cov_X, K=20)
# for i in range(50):
#     start = default_timer()
#     dp.update()
#     time1 = default_timer() - start

#     start = default_timer()
#     print(dp.elbo(), "time =", time1, default_timer() - start)