import numpy as np
from scipy.stats import binom, expon, poisson, gamma, norm

separation_line = "-"*79

n = 5
p = 0.6
cdf_binom = binom.cdf(1, n=n, p=p)
quantile_binom = binom.ppf(cdf_binom, n=n, p=p)
report_string_binom = (
    f"X~Binomial(n={n}, p={p}):\n"
    f"    P(X<=1) = {cdf_binom:.5f}\n"
    f"    Quantile({cdf_binom:.5f}) = {quantile_binom}"
)
print(separation_line)
print(report_string_binom)

lambda_ = 1
scale = 1/lambda_
quantile_lower_expon = expon.ppf(0.1, scale=scale)
quantile_higher_expon = expon.ppf(0.9, scale=scale)
report_string_expon = (
    f"X~Exp(lambda=1):\n"
    f"    For the middle 80% of the data:\n"
    f"    {quantile_lower_expon:.5f} <= X <= {quantile_higher_expon:.5f}"
)
print(separation_line)
print(report_string_expon)

mu = 3
pmf_poisson = poisson.pmf(1, mu=mu)
cdf_poisson = poisson.cdf(1, mu=mu)
complement_cdf_poisson = 1 - cdf_poisson
report_string_poisson = (
    f"X~Pois(Mu=3):\n"
    f"    P(X=1) = {pmf_poisson:.5f}\n"
    f"    P(X<=1) = {cdf_poisson:.5f}\n"
    f"    P(X>1) = {complement_cdf_poisson:.5f}\n"
)
print(separation_line)
print(report_string_poisson)

alpha = 2
beta = 1/3
scale = 1/beta
cdf_lower_gamma = gamma.cdf(0.5, a=alpha, scale=scale)
cdf_higher_gamma = gamma.cdf(1.5, a=alpha, scale=scale)
prob_gamma = cdf_higher_gamma - cdf_lower_gamma
report_string_gamma = (
    f"X~Gamma(alpha=2, beta=1/3):\n"
    f"    P(0.5<X<1.5) = {prob_gamma:.5f}\n"
)
print(separation_line)
print(report_string_gamma)

mu = 0
sigma = np.sqrt(1)
quantile_norm = norm.ppf(0.975, loc=mu, scale=sigma)
cdf_lower_norm = norm.cdf(-1.96, loc=mu, scale=sigma)
cdf_higher_norm = norm.cdf(1.96, loc=mu, scale=sigma)
prob_norm = cdf_higher_norm - cdf_lower_norm
quantile_symmetric_norm = norm.ppf(0.95, loc=mu, scale=sigma)
report_string_norm = (
    f"X~N(Mu={mu}, Sigma={sigma}):\n"
    f"    Quantile(0.975) = {quantile_norm:.5f}\n"
    f"    P(-1.96<=X<=1.96) = {prob_norm:.5f}\n"
    f"    Quantile(0.95) = {quantile_symmetric_norm:.5f}\n"
)
print(separation_line)
print(report_string_norm)