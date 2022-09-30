"""
Risk Based Optimization

"""

import cvxpy as cvx

from .exp import var_from_cov_matrix


def risk_budget(budget, cov):
    """
    risk budget model

    Reference: Bai, Scheinberg & Tutuncu (2013): Least-squares approach to risk parity in portfolio selection

    Parameters
    ----------
    budget: np.ndarray
        Risk budget \sum = 1
    cov: pd.DataFrame
        Variance covariance matrix

    Returns
    -------
    res
    """
    n = cov.shape[0]
    budget = budget.reshape((n, -1))

    weight = cvx.Variable(n, 'weight')
    budget_temp_var = cvx.Variable(n, name='temp_var_budget')
    var = var_from_cov_matrix(weight, cov)

    c = 100

    # Target Function
    object = var - budget @ budget_temp_var 

    # Logrithmic bounds
    



