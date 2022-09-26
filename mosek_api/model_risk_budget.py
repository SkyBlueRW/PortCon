"""
用于risk budgeting 与 risk parity的优化
"""
import gc 

import numpy as np 
import pandas as pd 

from mosek import fusion

from .base_optimizer import MosekOptimizer

def risk_budget(budget, cov):
    """
    risk budget model

    Reference: Bai, Scheinberg & Tutuncu (2013): Least-squares approach to risk parity in portfolio selection

    https://docs.mosek.com/MOSEKPortfolioCookbook-a4paper.pdf
    Parameters
    ----------
    budget: pd.Series
        风险预算 \sum = 1
    cov: pd.DataFrame
        方差协方差矩阵

    Returns
    -------
    res: pd.Series
        优化结果
    """
    Model = MosekOptimizer(
        "RiskBudget", budget.index.tolist(), long_only=False
    )
    t = Model.model.variable("BudgetTemp", len(budget))
    var = Model.func_variance("PortVariance", cov.values, Model.model.getVariable("x"))
    c = 100 

    # 目标函数
    Model.object_min(
        fusion.Expr.sub(
            var, fusion.Expr.dot(budget.values, t)
        ), "RiskBudget"
    )

    # Logrithmic bounds
    Model.model.constraint(
        "BudgetLogBound", fusion.Expr.hstack(
            Model.model.getVariable("x"), fusion.Expr.constTerm(len(budget), 1.), t
        ), fusion.Domain.inPExpCone()
    )

    Model.solve_model()

    # 归一
    res = Model.x()
    res = res / res.sum()

    Model.model.dispose()
    del Model 
    gc.collect()
    return res


def risk_parity(cov):
    """
    风险平价优化

    Parameter
    ---------
    cov: pd.DataFrame
        方差协方差矩阵
    
    Returns
    -------
    res: pd.Series
        风险平价的组合
    """
    n = len(cov)
    budget = pd.Series([1/n] * n, index=cov.index)
    return risk_budget(budget, cov)


def min_var(cov, long_only=True, budget=1, max_holding=None, min_holding=None):
    """
    最小方差组合
    """
    Model = MosekOptimizer(
        "MinVol", cov.index.tolist(), long_only=long_only
    )
    x = Model.model.getVariable("x")

    # 方差
    var = Model.func_variance("Var", cov.values, x)
    # 目标最小方差
    Model.object_min(var)
    Model.constraint_budget(budget)

    # 最大权重，最小权重
    if max_holding is not None:
        Model.constraint_max_holding(max_holding)
    if min_holding is not None:
        Model.constraint_min_holding(min_holding)
    
    # 解函数
    Model.solve_model()
    res = Model.x()

    Model.model.dispose()
    del Model
    gc.collect()
    return res 

def max_div(cov, long_only=True):
    """
    最大分散度组合
    """
    # 用于计算标准差的加权平均
    ret = pd.Series(np.diag(cov), index=cov.index) ** 0.5 

    Model = MosekOptimizer(
        "MaxDiv", cov.index.tolist(), long_only=long_only
    )
    x = Model.model.getVariable("x")
    # 方差
    var = Model.func_variance("Var", cov.values, x)
    # 目标最小方差
    Model.object_min(var)
    # 加权
    # Model.constraint_budget(1)
    Model.model.constraint("v", MosekOptimizer.func_weigted_sum(
        ret.values, x
    ), fusion.Domain.equalsTo(1.))

    Model.solve_model()
    res = Model.x()
    res /= res.sum()

    Model.model.dispose()
    del Model
    gc.collect()
    return res 



def max_div_vol(cov, max_std,
                max_holding=None, min_holding=None,
                budget=1, long_only=True):
    """
    最大分散度组合, 加入目标波动率版本
    """
    # 用于计算标准差的加权平均
    ret = pd.Series(np.diag(cov), index=cov.index) ** .5

    Model = MosekOptimizer(
        "MaxDivVol", cov.index.tolist(), long_only=long_only
    )
    x = Model.model.getVariable("x")

    # 目标函数
    Model.object_max(
        fusion.Expr.dot(ret.values, x), "MaxDivVol"
    )

    # 标准差约束
    Model.constraint_max_std(cov, max_std)
    # 预算约束
    Model.constraint_budget(budget)

    # 最大权重，最小权重
    if max_holding is not None:
        Model.constraint_max_holding(max_holding)
    if min_holding is not None:
        Model.constraint_min_holding(min_holding)
    
    # 解函数
    Model.solve_model()
    res = Model.x()

    Model.model.dispose()
    del Model
    gc.collect()
    return res 
    
    
