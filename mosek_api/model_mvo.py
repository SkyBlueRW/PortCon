"""
传统的mean variance 优化
"""
import gc 

from mosek import fusion

from .base_optimizer import MosekOptimizer

def minimum_variance(cov, active_risk=False,
    long_only=True, return_full_result=False,
    benchmark=None, budget=None, max_budget=None,
    max_holding=None, min_holding=None, max_active_holding=None, min_active_holding=None,
    industry=None, max_ind=None, min_ind=None, max_active_ind=None, min_active_ind=None,
    style=None, max_style=None, min_style=None, max_active_style=None, min_active_style=None,
    prev_holding=None, max_turnover=None):
    # 记录传入数据
    data = locals()
    data = {x: data[x] for x in data if data[x] is not None}

  
    name = "MinimumVariance"

    # 创建模型
    Model = MosekOptimizer.from_constraint(
        name, cov.index.tolist(), long_only=long_only, benchmark=benchmark, budget=budget, max_budget=max_budget,
        max_holding=max_holding, min_holding=min_holding, max_active_holding=max_active_holding, min_active_holding=min_active_holding,
        industry=industry, max_ind=max_ind, min_ind=min_ind, max_active_ind=max_active_ind, min_active_ind=min_active_ind,
        style=style, max_style=max_style, min_style=min_style, max_active_style=max_active_style, min_active_style=min_active_style,
        prev_holding=prev_holding, max_turnover=max_turnover
    )

    if active_risk:
        var_expr = Model.func_variance("ActiveVariance", cov.values,
                                       fusion.Expr.sub(Model.model.getVariable("x"), benchmark.values))
    else:
        var_expr = Model.func_variance("PortVariance", cov.values, Model.model.getVariable("x"))
    
    Model.object_min(var_expr, name)

    # 求解优化
    Model.solve_model()
    if return_full_result:
        res = Model.full_x()
        res.update({'data': data})
    else:
        res = Model.x()
    
    # 解除所有reference
    Model.model.dispose()
    del Model
    # garbage memory collecting
    gc.collect()


    return res

def mean_variance(
    alpha, risk_aversion, cov, active_risk=False, active_alpha=False,
    long_only=True, return_full_result=False,
    benchmark=None, budget=None, max_budget=None,
    max_holding=None, min_holding=None, max_active_holding=None, min_active_holding=None,
    industry=None, max_ind=None, min_ind=None, max_active_ind=None, min_active_ind=None,
    style=None, max_style=None, min_style=None, max_active_style=None, min_active_style=None,
    prev_holding=None, max_turnover=None
    ):
    """
    均值方差优化
    Max \alpha - risk_aversion * variance

    Parameters
    -----------
    alpha: pd.Series
        股票预期收益率
    risk_aversion: float
        风险厌恶系数
    active_alpha: bool
        若为True则最大化active alpha
    long_only: bool
        组合是否为long only组合
    return_full_result: bool
        若为True: 返回包含label与dual值的完整结果，可用于后续分析
        若为False: 则仅仅返回优化结果
    benchmark: pd.Series
        业绩基准，只有在提供业绩基准时各类active类的约束才可实现
    budget: float [0, 1.]
        组合budget，默认为1.
    max_holding: float
        组合单只股票最大持仓约束
    max_active_holding: float
        组合单只股票最大主动权重约束
    min_active_holding: float
        组合单只股票最小主动权重约束
    industry: pd.DataFrame
        index为行业名称，columns为股票代码的行业dummy variable
    max_ind: float
        组合最大行业暴露约束
    min_ind: float
        组合最小行业暴露约束
    max_active_ind: float
        组合最大主动行业暴露约束
    min_active_ind: float
        组合最小主动行业暴露约束
    style: pd.DataFrame
        index为style名称，columns为股票代码的style
    max_style: float
        组合最大风险因子暴露约束
    min_style: float
        组合最小风险因子暴露约束
    max_active_style: float
        组合最大主动风险因子暴露约束
    min_active_style: float
        组合最小主动风险因子暴露
    prev_holding: pd.Series
        前一期权重
    max_turnover: float
        最大换手约束，双边
    
    Returns
    ------
    full_result: 包含进行进一步分析的所有信息(优化结果与传入数据)
    x: 优化权重结果
    """
    # 记录传入数据
    data = locals()
    data = {x: data[x] for x in data if data[x] is not None}

    ret_label = "ActiveMean" if active_alpha else "Mean"
    cov_label = "ActiveVariance" if active_risk else "Variance"
    name = ret_label + cov_label

    # 创建模型
    Model = MosekOptimizer.from_constraint(
        name, alpha.index.tolist(), long_only=long_only, benchmark=benchmark, budget=budget, max_budget=max_budget,
        max_holding=max_holding, min_holding=min_holding, max_active_holding=max_active_holding, min_active_holding=min_active_holding,
        industry=industry, max_ind=max_ind, min_ind=min_ind, max_active_ind=max_active_ind, min_active_ind=min_active_ind,
        style=style, max_style=max_style, min_style=min_style, max_active_style=max_active_style, min_active_style=min_active_style,
        prev_holding=prev_holding, max_turnover=max_turnover
    )

    # 目标函数
    if active_alpha:
        ret_expr = fusion.Expr.dot(
            alpha.values, fusion.Expr.sub(Model.model.getVariable("x"), benchmark.values)
        )
    else:
        ret_expr = fusion.Expr.dot(
            alpha.values, Model.model.getVariable("x")
        )
    if active_risk:
        var_expr = Model.func_variance("ActiveVariance", cov.values,
                                       fusion.Expr.sub(Model.model.getVariable("x"), benchmark.values))
    else:
        var_expr = Model.func_variance("PortVariance", cov.values, Model.model.getVariable("x"))
    
    Model.object_max(
        fusion.Expr.sub(
            ret_expr, fusion.Expr.mul(risk_aversion, var_expr)
        ), name
    )

    # 求解优化
    Model.solve_model()
    if return_full_result:
        res = Model.full_x()
        res.update({'data': data})
    else:
        res = Model.x()
    
    # 解除所有reference
    Model.model.dispose()
    del Model
    # garbage memory collecting
    gc.collect()


    return res