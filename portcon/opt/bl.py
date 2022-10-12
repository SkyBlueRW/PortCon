"""
BL related function
"""

import numpy as np

def implied_ret(lbd, cov, weight):
    """
    weight implied expected return
    """
    return lbd * cov.dot(weight)


def posterior_mean(tau, cov, p, q, imp_ret):
    """
    expected value of expected return
    """
    inv_cov = np.linalg.inv(cov)

    p1 = np.linalg.inv(inv_cov/tau + p.dot(inv_cov).dot(p))
    p2 = (inv_cov/tau).dot(imp_ret) + p.dot(inv_cov).dot(q)
    return p1.dot(p2)