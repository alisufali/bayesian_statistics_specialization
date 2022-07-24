import numpy as np
import matplotlib.pyplot as plt


def likelihood_bernoulli(n, sum_x, theta):
    output = theta**(sum_x) * (1-theta)**(n-sum_x)
    return output


def log_likelihood_bernoulli(n, sum_x, theta):
    output = np.log(theta)*sum_x + np.log(1-theta)*(n-sum_x)
    return output


n = 400
sum_x = 72
thetas = np.linspace(start=0.01, stop=0.99, num=1000)
likelihoods = likelihood_bernoulli(n=n, sum_x=sum_x, theta=thetas)
log_likelihoods = log_likelihood_bernoulli(n=n, sum_x=sum_x, theta=thetas)

mle = sum_x/n

fig, ax = plt.subplots(ncols=2, figsize=(6.4*2.0, 4.8))
ax[0].plot(thetas, likelihoods)
ax[0].set_xlabel("Theta")
ax[0].set_ylabel("Likelihood")
ax[0].axvline(mle, color="black")
ax[1].plot(thetas, log_likelihoods)
ax[1].set_xlabel("Theta")
ax[1].set_ylabel("Log Likelihood")
ax[1].axvline(mle, color="black")

plt.show()
