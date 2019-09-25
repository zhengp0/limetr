# Mixed-effects model with linear random effects and trimming

LimeTr is a python package for fitting complex mixed-effects models with
noisy data. This specific branch is created to reproduce the results in the
corresponding paper.

## Installing
This packages require [conda](https://docs.conda.io/en/latest/) python
installation.
After cloning the repository change to the `paper` branch.
```
git clone https://github.com/zhengp0/limetr.git
git checkout paper
```
And then simply do.
```
make install
```
It will check if the following packages are installed,
* `numpy`,
* `scipy`,
* `ipopt`,

and install `limetr`.

## Simple Example
The general mathematical model can be found in the paper.
In order to run `limetr`, you need to specify,
* model for fixed-effects,
* model for random-effects,
* priors for fixed- and random-effects.

Data observation is needed and for the corresponding measurement standard
deviations, there are three options,
* provide measurement standard deviations for every observation,
* assume standard deviations are the same within the study and treat them
as variables,
* assume standard deviations are the same for all observations and treat it as
variable.

Here is a simple linear example,
```python
import numpy as np
import limetr


# define the dimensions
k_beta = 5
k_gamma = 1
num_obs = 100
num_studies = 3
study_sizes = np.array([20, 20, 60])

# simulate the data
# design matrices
X = np.random.randn(num_obs, k_beta)
Z = np.ones((num_obs, 1))
# true beta and gamma value
beta_true = np.random.randn(k_beta)
gamma_true = np.array([0.1]*k_gamma)
# create observations
S = np.array([0.05]*num_obs)
u = np.random.randn(num_studies, k_gamma)*np.sqrt(gamma_true)
U = np.repeat(u, study_sizes, axis=0)
obs_err = np.random.randn(num_obs)*S
obs_mean = X.dot(beta_true) + np.sum(Z*U, axis=1) + obs_err

# setup limetr model
def F(beta):
    return X.dot(beta)

def JF(beta):
    return X

lt = limetr.LimeTr(study_sizes,
                   k_beta,
                   k_gamma,
                   obs_mean,
                   F, JF,
                   Z, S=S,
                   inlier_percentage=0.95)

lt.fitModel()

# print solution
print("beta_true", beta_true)
print("beta_soln", lt.beta)
print("gamma_true", gamma_true)
print("gamma_soln", lt.gamma)
```