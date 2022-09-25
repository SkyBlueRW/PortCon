"""
CVXPY Expression
"""

import numpy as np 

import cvxpy as cp


def var_from_cov(weight, cov):
    """
    Calculate Portfolio Variance from var-cov matrix and weight

    $W^T Cov W$

    Parameters
    ---------
    weight: cp.Variable
        Portfolio weight
    cov: np.ndarry
        Variance Covariance matrix
    """
    return cp.quad_form(weight, cov)


def var_from_factor_model()
