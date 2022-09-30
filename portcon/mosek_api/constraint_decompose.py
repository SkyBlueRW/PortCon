"""
constraint decomposition
"""

import numpy as np 
import pandas as pd 

from . import utils 


def res_rhs_constraint_decompose(res_dict):
    """
    lhs = alpha + (-\sum(dual * derivative))的一阶Lagrangian分解
    当优化是均值方差优化时:
        lhs = V(x - b)或者 Vx
    当优化时限制最大标准差的最大收益率优化时:
        lhs = dual * Q (x - b)
    
    Parameters
    ----------
    res_dict: dict
        优化结果的 full result
    
    Returns
    -------
    res: dict 
        元素为 alpha与各个constraint block
    """
    res = {'alpha': res_dict['data']['alpha']}
    vars_res = utils.res_variable(res_dict)
    cons_res = utils.res_constraint(res_dict)

    # 不可导的约束不支持进行分解
    forbid_cons = ["MaxTurnover"]
    for i in forbid_cons:
        assert i not in cons_res.keys(), "不支持不可导的约束{}进行Lagrangian分解".format(i)
    
    res.update({
        "long_only": -vars_res["x"]['dual']
    })

    if "budget" in cons_res:
        res.update({
            "budget": pd.Series(- np.ones(len(vars_res['x'])) * cons_res['budget']['dual'], index=vars_res['x'].index)
        })
    
    for i in ("MaxPortHolding", "MaxActiveHolding", "MinActiveHolding"):
        if i in cons_res:
            _res = pd.Series(- np.eye(len(vars_res['x'])).dot(cons_res[i]['dual']), index=vars_res['x'].index)
            res.update({
                i: _res
            })
    
    for i in ("MaxInd", "MinInd", "MaxActiveInd", "MinActiveInd"):
        if i in cons_res:
            _res = - res_dict['data']['industry'].T.mul(cons_res[i]['dual'], axis=1)
            res.update({
                i: _res
            })
    
    for i in ("MaxStyle", "MinStyle", "MaxActiveStyle", "MinActiveStyle"):
        if i in cons_res:
            _res = -res_dict['data']['style'].T.mul(cons_res[i]['dual'], axis=1)
            res.update({
                i: _res
            })
    
    return res 


def implied_alpha_decompose(res_dict):
    """
    优化结果的implied alpha decomposition

    对于均值方差优化:
        V x^{\star} = \alpha + \sum \lambda_i a_i
        V (x^{\star} - b) = \alpha + \sum \lambda_i a_i
    对于有风险约束的最大收益率优化:
        需要注意的是，在这种优化下，只有在风险约束binding时才可以进行，否则该分解无意义
        \dfrac{\theta V x^{\star}}{RiskUpperBound} = \alpha + \sum \lambda_i a_i
        \dfrac{\theta V (x^{\star} - b)}{RiskUpperBound} = \alpha + \sum \lambda_i a_i

    Parameters
    ----------
    res_dict: dict 
        优化的full result
    
    Returns
    ------
    res: dict 
        分解后key 为implied alpha, alpha, 各constraint block的alpha modification的结果
    """
    res = res_rhs_constraint_decompose(res_dict)

    # 有风险约束的最大收益率优化
    if res_dict['objective_name'] in ("MaxAlpha", "MaxActiveAlpha"):
        # 主动风险约束
        if "MaxActiveStd" in res_dict["constraint_name"]:
            assert abs(res_dict['constraint_dual']['MaxActiveStd'][0]) > 0.00001, "给定风险约束最大化收益率类约束需要风险约束binding才可进行分解"
            lhs = res_dict['constraint_dual']['MaxActiveStd'][0] * res_dict['data']['cov'].values.dot(
                res_dict['variable_level']['x'] - res_dict['data']['benchmark'].values
            ) / res_dict['data']['max_active_std']    
        # 风险约束
        elif "MaxPortStd" in res_dict["constraint_name"]:
            assert abs(res_dict['constraint_dual']['MaxPortStd'][0]) > 0.00001, "给定风险约束最大化收益率类约束需要风险约束binding才可进行分解" 
            lhs = res_dict['constraint_dual']['MaxPortStd'][0] * res_dict['data']['cov'].values.dot(
                res_dict['variable_level']['x']
            ) / res_dict['data']['max_std']  
        else:
            raise TypeError("缺少标准差最大值")
        lhs = pd.Series(lhs, index=res_dict['variable_name']['x'])
        res.update({'implied_alpha': lhs})

    # 均值方差优化
    elif res_dict['objective_name'] in ("MeanVariance", "MeanActiveVariance", "ActiveMeanVariance", "ActiveMeanActiveVariance"):
        if res_dict['objective_name'] in ("MeanVariance", 'ActiveMeanVariance'):
            lhs = res_dict['data']['cov'].dot(res_dict['variable_level']['x']) * res_dict['data']['risk_aversion']
        elif res_dict['objective_name'] in ("MeanActiveVariance", 'ActiveMeanActiveVariance'):
            lhs = res_dict['data']['cov'].dot(res_dict['variable_level']['x'] - res_dict['data']['benchmark']) * res_dict['data']['risk_aversion']
        else:
            raise TypeError("错误的优化类型")    
        res.update({'implied_alpha': lhs})
    else:
        raise TypeError("当前不支持此类优化的约束分解")
    
    return res 


def holding_decompose(res_dict):
    """
    优化结果的 holding decomposition

    对于均值方差优化:

    对于有风险约束的最大收益率优化:

    Parameters
    ----------
    res_dict: dict 
        优化的 full result

    Returns
    -------
    res: dict 
        分解后key 为 x, optimal, 各constraint block的holding modification
    """
    res = res_rhs_constraint_decompose(res_dict)

    # 有风险约束的最大收益率优化
    if res_dict['objective_name'] in ("MaxAlpha", "MaxActiveAlpha"):
        # 主动风险约束
        if "MaxActiveStd" in res_dict["constraint_name"]:
            assert abs(res_dict['constraint_dual']['MaxActiveStd'][0]) > 0.00001, "给定风险约束最大化收益率类约束需要风险约束binding才可进行分解"
            v_inv = np.linalg.inv(res_dict['data']['cov'].values)
            v_inv = res_dict['data']['max_active_std'] / res_dict['constraint_dual']['MaxActiveStd'][0] * v_inv
            for i in res:
                if isinstance(res[i], pd.Series):
                    res[i] = pd.Series(v_inv.dot(res[i]), index=res[i].index)
                else:
                    res[i] = pd.DataFrame(v_inv.dot(res[i]), index=res[i].index, columns=res[i].columns) 
            res.update({"x": pd.Series(res_dict['variable_level']['x'], index=res_dict['variable_name']['x'])})
            res['x'].index.name = 'sid'
            res.update({"benchmark": res_dict['data']['benchmark']})
        elif "MaxPortStd" in res_dict['constraint_name']:
            assert abs(res_dict['constraint_dual']['MaxPortStd'][0]) > 0.00001, "给定风险约束最大化收益率类约束需要风险约束binding才可进行分解" 
            v_inv = np.linalg.inv(res_dict['data']['cov'].values)
            v_inv = res_dict['data']['max_std'] / res_dict['constraint_dual']['MaxPortStd'][0] * v_inv
            for i in res:
                if isinstance(res[i], pd.Series):
                    res[i] = pd.Series(v_inv.dot(res[i]), index=res[i].index)
                else:
                    res[i] = pd.DataFrame(v_inv.dot(res[i]), index=res[i].index, columns=res[i].columns) 
            res.update({"x": pd.Series(res_dict['variable_level']['x'], index=res_dict['variable_name']['x'])})
            res['x'].index.name = 'sid'
        else:
            raise TypeError("缺少标准差最大值")

    # 均值方差优化
    elif res_dict['objective_name'] in ("MeanVariance", "MeanActiveVariance", "ActiveMeanVariance", "ActiveMeanActiveVariance"):
        v_inv = np.linalg.inv(res_dict['data']['cov'].values) 
        v_inv = v_inv / res_dict['data']['risk_aversion']
        for i in res:
            if isinstance(res[i], pd.Series):
                res[i] = pd.Series(v_inv.dot(res[i]), index=res[i].index)
            else:
                res[i] = pd.DataFrame(v_inv.dot(res[i]), index=res[i].index, columns=res[i].columns) 
        res.update({"x": pd.Series(res_dict['variable_level']['x'], index=res_dict['variable_name']['x'])})
        res['x'].index.name = 'sid'
        if res_dict['objective_name'] in ("MeanActiveVariance", "ActiveMeanActiveVariance"):
            res.update({"benchmark": res_dict['data']['benchmark']})
    else:
        raise TypeError("当前不支持此类优化的约束分解")

    return res 


def return_attribute(ret, holding_dict, block_name=None):
    """
    给定holding分解的结果，进行return attribute

    Parameter
    ---------
    ret: pd.Series
        用于attribute的收益率，可以是realized也可以是估计的
    holding_dict: dict
        holding 分解的结果
    block_name: str 
        若不给定，则给出每个block的分解
        若给定，则给出每个block内部的分解
    
    Returns
    -------
    res: pd.Series
        收益分解的结果
    """
    if block_name is not None:
        assert block_name in holding_dict.keys(), "并不存在名为{}的holding block".format(block_name)
        res = holding_dict[block_name].mul(ret, axis=0).sum(axis=0)
    else:
        idx = []
        val = []
        for i in holding_dict.keys():
            if isinstance(holding_dict[i], pd.DataFrame):
                v = holding_dict[i].mul(ret, axis=0).sum(axis=0).sum()
            else:
                v = (holding_dict[i] * ret).sum()
            idx.append(i)
            val.append(v)
        res = pd.Series(val, index=idx)
    
    return res 
    

