"""
最大化IR

Reference: 
    1. Optimization Methods in Finance(2018) P103: 注模型给出的6.18有误。缺少了 \sum z = k 的必要约束
    2. Maximizing the Sharpe ratio and Information ratio in the Barra Optimizer
"""
import gc 

import numpy as np

from mosek import fusion

from .base_optimizer import MosekOptimizer




def max_ir(
    alpha, cov, max_holding=None, long_only=True
    ):
    """
    最大ir优化

    min z^T V z
        u^T z = 1
        Az - bk = 0
        Dz - dk < 0
        k>=0
    的formulation以在Ax=b, Dx<k的约束下进行ir最大的优化

    Parameters
    ---------
    alpha: pd.Series
        预期收益率/均值
    cov: pd.DataFrame
        方差协方差矩阵
    max_holding: scalar, list, np.ndarray
        最大权重上限
    long_only: bool
        是否允许负权重
    
    Returns
    ------
    res: pd.Series
        总和为1的权重

    """
    Model = MosekOptimizer(
        "MaxIR", alpha.index.tolist(), long_only=long_only
    )

    n = len(alpha)
    var = Model.func_variance("PortVariance", cov.values, Model.model.getVariable("x"))
    k = Model.model.variable('Scalar', 1, fusion.Domain.greaterThan(0.))

    # min z^T V z
    Model.object_min(var, "MaxIR")

    # u^T z = 1
    Model.model.constraint("alpha", MosekOptimizer.func_weigted_sum(alpha.values, Model.model.getVariable("x")), fusion.Domain.equalsTo(1.))
    Model.model.constraint("scale", fusion.Expr.sub(fusion.Expr.sum(Model.model.getVariable("x")), k), fusion.Domain.equalsTo(0.))

    if max_holding is not None:
        max_holding = max_holding if isinstance(max_holding, (list, np.ndarray)) else [max_holding] * n
        lhs = fusion.Expr.sub(
            MosekOptimizer.func_matrix_mul(np.eye(n), Model.model.getVariable("x")),
            fusion.Expr.mulElm(max_holding, fusion.Var.vstack([k]*n))
        )
        Model.model.constraint("MaxPortHolding", lhs, fusion.Domain.lessThan(0.))
    
    Model.solve_model()

    res = Model.x() / Model.x().sum()

    Model.model.dispose()
    del Model
    gc.collect()

    return res 



    






