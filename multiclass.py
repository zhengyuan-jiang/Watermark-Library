from __future__ import print_function
from statsmodels.stats.proportion import proportion_confint, multinomial_proportions_confint
import numpy as np
from scipy.stats import norm


def binary_search_compute_r(p_sn, p_ls, sigma_value, n):
    p_sn = min(p_sn, 1 - p_ls)
    lower_bound, upper_bound = 0.0, 10.0
    if p_ls <= p_sn / n:
        return 0.0
    while np.abs(lower_bound - upper_bound) > 1e-5:
        searched_radius = (lower_bound + upper_bound) / 2.0
        if norm.cdf(norm.ppf(p_ls) - searched_radius / sigma_value) >= norm.cdf(
                norm.ppf(p_sn) + searched_radius / sigma_value) / n:
            lower_bound = searched_radius
        else:
            upper_bound = searched_radius
    return lower_bound


def CertifyRadius(ls, probability_array, topk, sigma_value):
    p_ls = probability_array[ls]
    probability_array[ls] = -1
    sorted_index = np.argsort(probability_array)[::-1]
    sorted_probability_topk = probability_array[sorted_index[0:topk]]
    p_sk = np.zeros([topk], dtype=np.float64)
    radius_array = np.zeros([topk], dtype=np.float64)
    for i in np.arange(sorted_probability_topk.shape[0]):
        p_sk[0:i + 1] += sorted_probability_topk[i]
    for i in np.arange(topk):
        radius_array[i] = binary_search_compute_r(p_sk[i], p_ls, sigma_value, topk - i)
    return np.amax(radius_array)


def multi_ci(counts, alpha):
    multi_list = []
    n = np.sum(counts)
    l = len(counts)
    for i in range(l):
        multi_list.append(
            proportion_confint(min(max(counts[i], 1e-10), n - 1e-10), n, alpha=alpha * 2. / l, method="beta"))
    return np.array(multi_list)


def compute_r(data, args):
    r_array = np.empty(data.shape[0])
    for idx in range(data.shape[0]):
        ls = int(data[idx][-1])

        class_freq = data[idx][:-1]
        CI = multi_ci(class_freq, args.alpha)
        pABar = CI[ls][0]
        probability_bar = CI[:, 1]
        probability_bar = np.clip(probability_bar, a_min=-1, a_max=1 - pABar)
        probability_bar[ls] = pABar
        r = CertifyRadius(ls, probability_bar, args.k, args.sigma)
        r_array[idx] = r

        # print("{}\t{:.3}".format(idx + 1, r), flush=True)

    return r_array