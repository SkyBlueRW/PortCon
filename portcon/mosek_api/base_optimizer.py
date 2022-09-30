"""
Low level optimization solver object
"""
from copy import deepcopy

import numpy as np 
import pandas as pd 
from mosek import fusion

from . import utils


class MosekOptimizer:
    """
    Holder of optimization solver

    Model General Formulation
    -------------------------
    对原Mosek fusion进行wrap后的MosekOptimizer， 建模的形式应当为
        min/max f(x)
            s.t. h(x) <= 0
                 g(x) = 0
    其中：
        1. 函数f, h, g是通过func_*类的各类函数实现。得到的f(x), h(x), g(x)通常是以fusion.Expr, fusion.Variable的形式存在，再分别
        2. 调用object_max/min 设置 目标函数
        3. 调用各类constraint_函数增添约束

    Mosek: 在Mosek fusion的建模中，所有的优化都以一下的函数形式存在。所有的约束的lhs都是以线性的方式存在
        min c*x
            s.t. Ax + b belongs_to K
        其中k为一个set (或者conic, linear, semidefinite, binary, ...etc)，详见add_constraint与add_variable的函数说明 
        左侧的 Ax + b 以fusion.Expr的形式表示， 也可以是变量本身(A=I, b=0)
    """

    def __init__(self, name, asset_ids, long_only=True):
        """
        初始化mosek solver 并创建求解变量

        Parameters
        ---------
        name: str
            优化器名称
        asset_ids: list, np.ndarray 
            优化问题资产ID (代码或名称)
        """
        # 初始化mosek solver
        self.model = fusion.Model(name)
        # 增添 求解变量
        if long_only:
            self.model.variable("x", len(asset_ids), fusion.Domain.greaterThan(0.))
        else:
            self.model.variable("x", len(asset_ids))
        # 记录目标函数名称
        self._objective_name = None 
        # 记录约束条件名称, 新增约束应该在此注册。若约束以block形式存在(多个同一类型)则为list of str，反之则为str
        # 注意 此处只记录在约束分解中有意义的约束，以便于对结果进行 constraint decomposition
        self._constraint_name = []
        # 记录变量名称, 所有的新增变量应该在此注册。若变量以block形式存在(多个同一类型)则为list of str, 反之则为str
        self._variable_name = [["x"]+list(asset_ids)]
        # 优化问题当前状态
        self.status = 'Unsolved'


    def __repr__(self):
        """
        instance描述 
        """
        descript = """Objective: {0}\rVariable Block: {1}\rConstraint Block: {2} """.format(
                          self.get_objective_name(),
                          ", ".join(self.get_variable_name(block_name=None)),
                          ", ".join(self.get_constraint_name())
                      )
        return descript

    # ---------------- 工具函数 -------------------
    @staticmethod
    def get_size(expr):
        """
        获取表达式维度

        Parameters
        ----------
        expr: fusion.Variable, fusion.Expr, fusion.Constraint, np.ndarray, fusion.Matrix

        Returns
        -------
        tuple or int
            表达式对应的维度/长度
            若对应的数据为2维度则返回tuple, 若数据为1维则返回int
        """
        if isinstance(expr, (fusion.Variable, fusion.Expr, fusion.ExprReshape, 
                             fusion.Constraint, fusion.ExprAdd)):
            return expr.getSize()
        elif isinstance(expr, (np.ndarray, pd.DataFrame)):
            n = expr.shape
            n = n[0] if len(n) == 1 else n 
            return n
        elif isinstance(expr, list):
            return len(expr)
        elif isinstance(expr, fusion.Matrix):
            return (expr.numRows(), expr.numColumns())
        else:
            raise TypeError("未能识别的expr变量类型")

    # ---------------- labeling 相关 --------------------
     
    def get_constraint_name(self, block_name=None):
        """
        返回constraint的名称

        Parameters
        ---------
        block_name: str, optional
            constraint block的名称
        
        Returns
        -------
        res: list / str 
            当不提供block_name时，返回每个constraint block的名称
                i.e: ['long_only', 'budget']
            当提供block_name时，返回该constraint block中所有的约束的labeling
                i.e: 'long_only' --> ['000001', '000002']
        """
        res = [x[0] if isinstance(x, list) else x for x in self._constraint_name]
        if block_name is None:
            return res 
        else:
            if block_name not in res:
                return None
            else:
                res = deepcopy(self._constraint_name[res.index(block_name)])
                res = None if isinstance(res, str) else res[1: ] 
                return res 

    def get_variable_name(self, block_name="x"):
        """
        返回variable的名称

        Parameters
        ---------
        block_name: str, optional
            variable block的名称
        
        Returns
        -------
        res: list / str 
            当不提供block_name时，返回每个variable block的名称
                i.e: ['x', 'buy_turnover', 'sell_turnover']
            当提供block_name时，返回该variable block中所有的约束的labeling
                i.e: 'x' --> ['000001', '000002']
        """
        res = [x[0] if isinstance(x, list) else x for x in self._variable_name]
        if block_name is None:
            return res 
        else:
            if block_name not in res:
                return None
            else:
                res = deepcopy(self._variable_name[res.index(block_name)])
                res = None if isinstance(res, str) else res[1: ] 
                return res

    
    def get_objective_name(self):
        """目标函数名称"""
        return deepcopy(self._objective_name)

    # -----------------  目标函数相关 -------------------

    def object_max(self, expr, name=None):
        """
        设置目标函数 - 最大

        Parameters
        ---------
        expr: fusion.Expr / fusion.Variable (must evaluates to scalar, 即只支持single objective 优化)
            目标函数
        """
        self.model.objective(fusion.ObjectiveSense.Maximize, expr)
        self._objective_name = name if name is not None else "Unnamed Max Objective"

    def object_min(self, expr, name=None):
        """
        设置目标函数 - 最小

        Parameters
        ---------
        expr: fusion.Expr / fusion.Variable (must evaluates to scalar, 即只支持single objective 优化)
            目标函数
        """
        self.model.objective(fusion.ObjectiveSense.Minimize, expr)
        self._objective_name = name if name is not None else "Unnamed Min Objective"

    # ----------------- func -----------------
    def func_std(self, name, cov, expr):
        """
        给定expr以及cov，返回对应的std的fusion.Variable

        Parameters
        ----------
        name: str
            返回的变量对应的名称
        cov: np.ndarray
            协方差矩阵
        expr: fusion.Expr / fusion.Variable
            权重，主动权重等表达式
        
        Returns
        -------
        res: fusion.Variable
            res ** 2 >= expr^T cov expr = exp^T G G^T expr 这是一个n+1 dimension的约束
        """
        # 创建对应dummy variable
        std = self.model.variable(name, 1, fusion.Domain.unbounded())
        self._variable_name.append(name)
        # 通过约束条件获得 对应std的dummy variable
        g = np.linalg.cholesky(cov)
        std_con = self.model.constraint(name,
            fusion.Expr.vstack(std, fusion.Expr.mul(
                    fusion.Matrix.dense(np.transpose(g)), expr)
            ), fusion.Domain.inQCone()
        )
        return std 

    def func_variance(self, name, cov, expr):
        """
        给定expr以及cov，返回对应的variance

        Parameters
        ----------
        name: str
            返回的变量对应的名称
        cov: np.ndarray
            协方差矩阵
        expr: fusion.Expr / fusion.Variable
            权重，主动权重等表达式
        
        Returns
        -------
        res: fusion.Variable
            res >= expr^T cov expr = exp^T G G^T expr 这是一个n+2 dimension的约束
        """

        # 创建对应dummy variable
        var = self.model.variable(name, 1, fusion.Domain.unbounded())
        self._variable_name.append(name)
        # 通过约束条件获得对应variance的dummy variable
        g = np.linalg.cholesky(cov)
        var_con = self.model.constraint(name,
            fusion.Expr.vstack(0.5, var, fusion.Expr.mul(
                fusion.Matrix.dense(np.transpose(g)), expr
                )
            ), fusion.Domain.inRotatedQCone()
        ) 
        return var

    def func_abs(self, name, expr):
        """
        给定expr，返回对应的绝对值

        Parameters
        ----------
        name: name 
            返回的变量对应的名称
        expr: fusion.Expr / fusion.Variable
            权重，基准偏离等
        
        Returns
        ------
        res: fusion.Variable
            res >= |expr| 这是一个n维度的约束
        """
        # 创建对应abs(expr)的dummy variable
        abs_var = self.model.variable(name, MosekOptimizer.get_size(expr), fusion.Domain.unbounded())
        self._variable_name.append(name)
        # 通过约束条件获得对应abs的dummy variable
        long_side_con = self.model.constraint("{}_long".format(name),
            fusion.Expr.sub(abs_var, expr), fusion.Domain.greaterThan(0.)
        )
        short_side_con = self.model.constraint("{}_short".format(name),
            fusion.Expr.add(abs_var, expr), fusion.Domain.greaterThan(0.)
        )

        return abs_var

    @staticmethod
    def func_weigted_sum(weight, expr):
        """
        返回加权平均 weight^T expr

        Parameters
        ---------
        weight: np.ndarray / list
            权重
        expr: fusion.Expr, fusion.variable
            对应表达式
        
        Returns
        -------
        res: fusion.Expr
        """
        return fusion.Expr.dot(weight, expr)
    
    @staticmethod
    def func_matrix_mul(A, expr):
        """
        返回矩阵乘法结果  A * expr

        Parameters
        ----------
        A: np.ndarray
            左乘矩阵
        expr: fusion.Expr / fusion.Variable
            权重等
        
        Return
        -----
        res: fusion.Expr
            对应A * expr的矩阵乘法结果
        """

        # 注： 此处需要astype的原因是dense只接收float32, int32而int往往被优化为int8
        return fusion.Expr.mul(
            fusion.Matrix.dense(A.astype(float)), expr 
        )
        # return fusion.Expr.mul(
        #     A, expr
        # )
    
    @staticmethod
    def func_eye_matrix_mul(expr):
        """
        返回expr的每个元素为一列 I * expr

        Parameters
        ----------
        expr: fusion.Expr / fusion.Variable
            权重等
        
        Returns
        -------
        res: fusion.Expr
            对应I * expr的矩阵乘法结果，用于限制expr中的每个元素
        """
        return MosekOptimizer.func_matrix_mul(
            np.eye(MosekOptimizer.get_size(expr)), expr
        )


    # --------------- 约束 ------------------------
    def constraint_linear(self, name, A, expr, rhs, operation):
        """
        添加 A * expr <=> rhs 类型的线性约束

        Parameters
        ----------
        name: str 
            约束block的名称
        A: pd.DataFrame
            做成矩阵，每一列对应一个约束
        expr: fusion.Expr, fusion.Variable
            权重等
        rhs: list, float
            right hand side 
        operation: str
            <, =, >
        
        """
        assert name not in self.get_constraint_name(), "已存在名为{}的constraint block".format(name)
        
        lhs = self.func_matrix_mul(A.values, expr)
        if operation in ("<=", "<"):
            op = fusion.Domain.lessThan(rhs)
        elif operation in ("="):
            op = fusion.Domain.equalsTo(rhs)
        elif operation in (">=", ">"):
            op = fusion.Domain.greaterThan(rhs)
        else:
            raise TypeError("当前只支持<,=,>三种运算")
        cons = self.model.constraint(name, lhs, op)

        # 添加label
        self._constraint_name.append([name] + A.index.tolist())
    
    def constraint_budget(self, rhs=1.):
        """
        添加 \sum x = rhs的budget约束

        Parameters
        ----------
        rhs: float
            \ sum x = rhs

        """
        assert "budget" not in self.get_constraint_name(), "已存在名为budget的constraint block"

        self.model.constraint("budget", fusion.Expr.sum(self.model.getVariable("x")),
                        fusion.Domain.equalsTo(rhs)
        )
        self._constraint_name.append("budget")

    def constraint_max_budget(self, rhs=1.):
        """
        添加 \sum x <= rhs的budget约束

        Parameters
        ----------
        rhs: float
            \ sum x <= rhs
        """
        assert "MaxBudget" not in self.get_constraint_name(), "已存在名为MaxBudget的constraint block"

        self.model.constraint("MaxBudget", fusion.Expr.sum(self.model.getVariable("x")),
                        fusion.Domain.lessThan(rhs)
        )
        self._constraint_name.append("MaxBudget")


    def constraint_max_std(self, cov, ths, bench=None):
        """
        添加最大标准差约束
        当提供bench时，为主动标准差约束 x
        当不提供bench时，为组合标准差约束 x-x_b

        Parameters
        ----------
        cov: pd.DataFrame
            协方差矩阵
        ths: float
            最大标准差值
        bench: pd.Series, optional
            业绩基准
        """
        constraint_label = "MaxPortStd" if bench is None else "MaxActiveStd"
        assert constraint_label not in self.get_constraint_name(), "已存在{}约束".format(constraint_label)

        x = self.model.getVariable("x") if bench is None \
            else fusion.Expr.sub(self.model.getVariable("x"), bench.values)
        # 构建标准差变量
        std = self.func_std(constraint_label[3:], cov.values, x)
        # 创建标准差约束
        self.model.constraint(constraint_label, std, fusion.Domain.lessThan(ths))
        self._constraint_name.append(constraint_label)

    def constraint_max_holding(self, ths, bench=None):
        """
        添加最大持仓权重
        当提供bench时，为主动持仓约束 x
        当不提供bench时，为组合持仓约束 x-x_b

        Parameters
        ----------
        cov: pd.DataFrame
            协方差矩阵
        ths: float
            最大标准差值
        bench: pd.Series, optional
            业绩基准
        """
        # func_eye_matrix_mul
        constraint_label = "MaxPortHolding" if bench is None else "MaxActiveHolding"
        assert constraint_label not in self.get_constraint_name(), "已存在{}约束".format(constraint_label)  

        x = self.model.getVariable("x") if bench is None else fusion.Expr.sub(self.model.getVariable("x"), bench.values)
        cons = self.model.constraint(constraint_label, MosekOptimizer.func_eye_matrix_mul(x), fusion.Domain.lessThan(ths))
        self._constraint_name.append([constraint_label] + self.get_variable_name('x'))

    def constraint_min_holding(self, ths, bench=None):
        """
        添加最小持仓权重
        当提供bench时，为主动持仓约束 x
        当不提供bench时，为组合持仓约束 x-x_b

        Parameters
        ----------
        cov: pd.DataFrame
            协方差矩阵
        ths: float
            最大标准差值
        bench: pd.Series, optional
            业绩基准
        """
        # func_eye_matrix_mul
        constraint_label = "MinPortHolding" if bench is None else "MinActiveHolding"
        assert constraint_label not in self.get_constraint_name(), "已存在{}约束".format(constraint_label)  

        x = self.model.getVariable("x") if bench is None \
            else fusion.Expr.sub(self.model.getVariable("x"), bench.values)
        cons = self.model.constraint(constraint_label, MosekOptimizer.func_eye_matrix_mul(x), fusion.Domain.greaterThan(ths))
        self._constraint_name.append([constraint_label] + self.get_variable_name('x'))
    

    def constraint_max_turnover(self, prev_holding, ths):
        """
        添加最大换手约束(双边换手)

        Parameters
        ----------
        prev_holding: pd.Series
            前一期权重
        ths: float
            最大换手
        """
        assert "MaxTurnover" not in self.get_constraint_name(), "已存在{}约束".format("MaxTurnover")

        abs_v = self.func_abs(
            "turnover", fusion.Expr.sub(self.model.getVariable("x"), prev_holding.values)
        )
        # 为turnover variable block每个单独的变量增添label
        self._variable_name = [["turnover"]+self.get_variable_name("x") if x=='turnover' else x for x in self._variable_name ]
        cons = self.model.constraint("MaxTurnover", fusion.Expr.sum(abs_v), fusion.Domain.lessThan(ths))
        self._constraint_name.append("MaxTurnover")



    # ------------------ 求解 -----------------
    def solve_model(self):
        """
        应当接受可能的param(优化精度等)
        进行优化求解并更新优化问题的status
        """
        try:
            self.model.solve()
            self.model.acceptedSolutionStatus(fusion.AccSolutionStatus.Optimal)
            self.status = 'Success'
        except fusion.OptimizeError as e:
            self.status = e 
        except fusion.SolutionError as e:
            # 分析原因
            prosta = self.getProblemStatus()
            if prosta == fusion.ProblemStatus.DualInfeasible:
                self.status = 'Dual Infeasibility'
            elif prosta == fusion.ProblemStatus.PrimalInfeasible:
                self.status = 'Primal Infeasibility'
            else:
                self.status = 'Failed'
        except Exception as e:
            self.status = 'Failed'

    # ---------------- 结果输出 ----------------
    def full_x(self):
        """
        返回完整的优化结果
        """
        res = dict()
        res.update({'status': self.status})

        res.update({'objective_name': self._objective_name})
        # variable 
        var_name = dict()
        var_level = dict()
        var_dual = dict()
        for name in self.get_variable_name(block_name=None):
            var_name.update({
                name: self.get_variable_name(name)
            })
            var_level.update({
                name: list(self.model.getVariable(name).level())
            })
            var_dual.update({
                name: list(self.model.getVariable(name).dual())
            })
        res.update({'variable_name': var_name})
        res.update({'variable_level': var_level})
        res.update({'variable_dual': var_dual})

        # constraint 
        con_name = dict()
        con_level = dict()
        con_dual = dict()
        for name in self.get_constraint_name(block_name=None):
            con_name.update({
                name: self.get_constraint_name(name)
            })
            con_level.update({
                name: list(self.model.getConstraint(name).level())
            })
            con_dual.update({
                name: list(self.model.getConstraint(name).dual())
            })
        res.update({'constraint_name': con_name})
        res.update({'constraint_level': con_level})
        res.update({'constraint_dual': con_dual})

        return res 

    def x(self):
        """
        返回组合权重的优化结果

        Returns
        -------
        res: pd.Series
            组合权重优化结果
        """
        if self.status == 'Success':
            return pd.Series(self.model.getVariable("x").level(), index=self.get_variable_name("x"))
        elif self.status == 'Unsolved':
            raise NotImplementedError("请先solve_model")
        else:
            raise ValueError("求解错误: {}".format(self.status))

    def var_res(self, var_block):
        """
        返回变量值与dual value

        Parameter
        ---------
        res_dict: dict 
            优化结果的full res
        variable_block: str
            variable block的名称

        Returns
        -------
        当为多维度的variable block时返回pd.DataFrame
        当为单维度的单一variable时返回pd.Series
        若不给定variable block 则返回dict
        """
        res = self.full_x()
        return utils.res_variable(res, var_block)

    def con_res(self, con_block):
        """
        返回变量值与dual value

        Parameter
        ---------
        res_dict: dict 
            优化结果的full res
        constraint_block: str
            constraint block的名称

        Returns
        -------
        当为多维度的constraint block时返回pd.DataFrame
        当为单维度的单一constraint时返回pd.Series
        若不给定constraint block 则返回dict
        """
        res = self.full_x()
        return utils.res_constraint(res, con_block)

    # ----------------- factory ---------------

    @classmethod
    def from_constraint(
        cls, name, universe, long_only=True, budget=None, max_budget=None,
        benchmark=None, 
        alpha=None, min_alpha=None, min_active_alpha=None, 
        cov=None, max_std=None, max_active_std=None,
        max_holding=None, min_holding=None, max_active_holding=None, min_active_holding=None,
        industry=None, max_ind=None, min_ind=None, max_active_ind=None, min_active_ind=None,
        style=None, max_style=None, min_style=None, max_active_style=None, min_active_style=None,
        prev_holding=None, max_turnover=None
    ):
        """
        用于标准化各Model约束条件的标准化输入
        返回一个已经添加这些约束的MosekOptimizer instance

        Parameters
        ----------
        name: str
            模型名称
        universe: list of str
            选股股票池的list
        long_only: bool
            组合是否为long only组合
        benchmark: pd.Series
            业绩基准，只有在提供业绩基准时各类active类的约束才可实现
        budget: float [0, 1.]
            组合budget，默认为1.
        alpha: pd.Series
            预期收益率预测
        min_alpha: float
            最小预期收益率约束
        min_active_alpha: float
            最小主动预期收益率约束
        cov: pd.DataFrame
            股票协方差矩阵
        max_std: float
            组合最大标准差约束
        max_active_std: float
            组合最大主动标准差约束
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
        -------
        Model: MosekOptimizer instance
            已经添加了这些约束的instance
        """
        # 数据检查
        if benchmark is None:
            assert (max_active_std is None) &\
                (max_active_holding is None) & (min_active_holding is None) &\
                (max_active_ind is None) & (min_active_ind is None) & \
                (max_active_style is None) & (min_active_style is None), "提供benchmark数据以进行active类的约束"
        if cov is None:
            assert (max_std is None) & (max_active_std is None), "提供cov数据以进行std类的约束"
        if industry is None:
            assert (max_ind is None) & (min_ind is None) &\
                (max_active_ind is None) & (min_active_ind is None), "提供industry数据以进行ind类的约束"
        if style is None:
            assert (max_style is None) & (min_style is None) &\
                (max_active_style is None) & (min_active_style is None), "提供style数据以进行style类的约束"
        if prev_holding is None:
            assert max_turnover is None, "提供prev_holding以进行turnover类约束"

        # 初始化模型
        Model = cls(name, universe, long_only)

        # budget 约束
        if budget is not None:
            Model.constraint_budget(budget)
        
        if max_budget is not None:
            Model.constraint_max_budget(max_budget)
        
        # 最小预期收益率
        if min_alpha is not None:
            Model.model.constraint(
                "MinAlpha", fusion.Expr.dot(alpha.values, Model.model.getVariable("x")),
                fusion.Domain.greaterThan(min_alpha)
                )
        if min_active_alpha is not None:
            Model.model.constraint(
                "MinActiveAlpha", fusion.Expr.dot(alpha.values, 
                fusion.Expr.sub(Model.model.getVariable("x"), benchmark.values)),
                fusion.Domain.greaterThan(min_active_holding)
            )

        # 最大标准差约束
        if max_std is not None:
            Model.constraint_max_std(cov, max_std)
        if max_active_std is not None:
            Model.constraint_max_std(cov, max_active_std, bench=benchmark)
        
        # 最大最小权重约束
        if max_holding is not None:
            Model.constraint_max_holding(max_holding)
        if min_holding is not None:
            Model.constraint_min_holding(min_holding)
        if max_active_holding is not None:
            Model.constraint_max_holding(max_active_holding, benchmark)
        if min_active_holding is not None:
            Model.constraint_min_holding(min_active_holding, benchmark)

        # 行业约束
        if max_ind is not None:
            Model.constraint_linear("MaxInd", industry, Model.model.getVariable("x"), max_ind, "<")
        if min_ind is not None:
            Model.constraint_linear("MinInd", industry, Model.model.getVariable("x"), min_ind, ">")
        if max_active_ind is not None:
            Model.constraint_linear("MaxActiveInd", industry, 

                                    fusion.Expr.sub(Model.model.getVariable("x"), benchmark.values), max_active_ind, "<"
            )
        if min_active_ind is not None:
            Model.constraint_linear("MinActiveInd", industry,
                                    fusion.Expr.sub(Model.model.getVariable("x"), benchmark.values), min_active_ind, ">"
            )
        
        # 风险因子约束
        if max_style is not None:
            Model.constraint_linear("MaxStyle", style, Model.model.getVariable("x"), max_style, "<")
        if min_style is not None:
            Model.constraint_linear("MinStyle", style, Model.model.getVariable("x"), min_style, ">")
        if max_active_style is not None:
            Model.constraint_linear("MaxActiveStyle", style,
                                    fusion.Expr.sub(Model.model.getVariable("x"), benchmark.values), max_active_style, "<"
            )
        if min_active_style is not None:
            Model.constraint_linear("MinActiveStyle", style,
                                    fusion.Expr.sub(Model.model.getVariable("x"), benchmark.values), min_active_style, ">"
            )
        
        # 最大换手约束
        if max_turnover is not None:
            Model.constraint_max_turnover(prev_holding, max_turnover)

        return Model
