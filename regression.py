from scipy.stats import binom
from scipy.stats import norm
import math


def certify(epsilon, args):
    sigma = args.sigma
    c = 1 - args.alpha

    p_lower = norm.cdf(-1 * (epsilon / sigma))
    upper_bound_of_klower = math.floor(args.num_noise * p_lower)

    if upper_bound_of_klower == 0:
        return -1

    for i in range(upper_bound_of_klower):

        c_ = 1 - binom.cdf(i, args.num_noise, p_lower)
        if c_ < c:
            break

    if i == upper_bound_of_klower:
        return -1

    return (i - 1)


def find_index(N, v):
    left, right = 0, len(N) - 1
    while left <= right:
        mid = (left + right) // 2
        if N[mid] == v:
            return mid
        elif N[mid] < v:
            right = mid - 1
        else:
            left = mid + 1
    return left