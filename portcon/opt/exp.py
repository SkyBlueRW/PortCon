"""
Expression for risk related component

https://www.cvxpy.org/tutorial/functions/index.html
"""

from re import L
import numpy as np 

import cvxpy as cvx

__all__ = [
    # Risk Measure
    'var_from_cov_matrix', 'idio_var_from_idio_std', 'sys_var_from_factor_cov', 'var_from_factor_cov'

    # Transaction
]


def var_from_cov_matrix(wgt, cov):
    """
    Calculate Portfolio Variance from var-cov matrix and weight
    $W^T Cov W$
    
    Parameters
    ---------
    weight: cp.Variable
        Portfolio weight or active/ rebalancing weight
    cov: np.ndarry
        Variance Covariance matrix

    Returns expression
    """

    return cvx.quad_form(
        wgt, cov
    )

def ido_var_from_idio_std(wgt, idio_std):
    """
    Calculate idiosyncratic variance from idiosyncratic std vector
    """
    return cvx.sum_squares(
        idio_std @ wgt
    )


def sys_var_from_factor_cov(wgt, exposure, factor_cov):
    """
    Calculate systematic variance from factor covariance
    """
    return cvx.quad_form(
        (wgt.T @ exposure).T, factor_cov
    )

def var_from_factor_cov(wgt, exposure, factor_cov, idio_std):
    """
    Calculate portfolio variance from factor model

    Paramters
    -------
    weight: cp.Variable
        Portfolio weight or active/ rebalancing weight
    exposure: np.ndarry
        n * p
    factor_cov: np.ndarray
        factor covariance matrix p * p
    idio_std: np.ndarray
        idiosyncratic std: n * n

    Returns
    -------
    expression for portfolio variance 
    """
    return ido_var_from_idio_std(wgt, idio_std) + sys_var_from_factor_cov(wgt, exposure, factor_cov)