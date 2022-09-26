"""
特征组合
"""

import numpy as np 

def simple_minvar(A, b, cov):
    """
    给定约束的最小方差组合解析解
    Min x^T cov x
        s.t Ax = b

    Parameters
    ----------
    A: pd.DataFrame
        sid * char
    b: pd.Series, np.ndarray
        右侧约束
    cov: pd.DataFrame
        协方差矩阵
    
    """
    b = b.reindex(A.columns).values if isinstance(b, pd.Series) else b 
    z = np.linalg.solve(cov, A)
    res = np.dot(z, np.linalg.solve(A.T.dot(z), b))
    return pd.Series(res, index=A.index)


