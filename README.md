# DD2434-project

## sklearn.mixture.BayesianGaussianMixture

BGM in sklearn uses Gaussian-Wishart distribution for each components, while we will use Gaussian-Gaussian distribution. That is, Sklearn assumes the variance in each component is not fixed and follows a distribution. In contrast, we assume a fixed variance in each component. Plus, sklearn omitts constant terms when computing ELBO.