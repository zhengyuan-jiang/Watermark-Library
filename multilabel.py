import numpy as np
from statsmodels.stats.proportion import proportion_confint, multinomial_proportions_confint
from scipy.stats import norm, binom_test


def multi_ci(counts, alpha):  # [0] lower [1] upper return (n,2)  counts, alpha confidence
    multi_list = []
    n = np.sum(counts)
    l = len(counts)
    counts = np.clip(counts, 1e-10, n - 1e-10)
    for i in range(l):
        multi_list.append(proportion_confint(counts[i], n, alpha=alpha * 2. / l, method="beta"))
    return np.array(multi_list)


def reverse_sort(d):
    d = np.sort(-d)
    return -d


def sum_(p, l, r):  # l,r starts from 1 ,but numpy indices start from 0
    ret = 0
    for i in range(l - 1, r):
        ret = ret + p[i]
    return ret


def get_overlap(lower_p, upper_p, L, k, k_prime, r, sigma):  # k smoothed output kprime hard output
    d = int(sum(L))
    lower_p = np.multiply(lower_p, L)
    upper_p = np.multiply(upper_p, 1 - L)
    lower_p = reverse_sort(lower_p)
    upper_p = reverse_sort(upper_p)
    for e in range(min(d, k), 0, -1):
        s = k - e + 1
        LHS = norm.cdf(norm.ppf(lower_p[e - 1]) - r / sigma)
        RHS = norm.cdf(norm.ppf(upper_p[s - 1]) + r / sigma)
        eta = d - e + 1
        for u in range(1, eta + 1):
            pAu = sum_(lower_p, e, e + u - 1)
            val = norm.cdf(norm.ppf(pAu / k_prime) - r / sigma)
            LHS = max(LHS, k_prime / u * val)
        for v in range(1, s + 1):
            pBv = sum_(upper_p, s - v + 1, s)
            val = norm.cdf(norm.ppf(pBv / k_prime) + r / sigma)
            RHS = min(RHS, k_prime / v * val)
        if LHS > RHS:
            return e
    return 0


def Get_Overlap(counts, target, k, k_prime, alpha, perturbation, sigma):
    P_L_U = multi_ci(counts, alpha).transpose()
    e = get_overlap(P_L_U[0], P_L_U[1], target, k, k_prime, perturbation, sigma)
    return e