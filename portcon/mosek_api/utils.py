"""
组合构建功能函数
"""

import pandas as pd 



def res_variable(res_dict, variable_block=None):
    """
    对full_result中的variable部分进行formation,
    返回其变量值与dual value

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
    def _res_variable(res_dict, variable_block):
        assert variable_block in res_dict['variable_name'].keys(), "不存在名为{}的variable block".format(variable_block)
        if isinstance(res_dict['variable_name'][variable_block], list):
            df = pd.DataFrame(
                    {'level': res_dict['variable_level'][variable_block],
                    'dual': res_dict['variable_dual'][variable_block]
                    }, index=res_dict['variable_name'][variable_block]
                )
        else:
            df = pd.Series([res_dict['variable_level'][variable_block][0],
                            res_dict['variable_dual'][variable_block][0]],
                            index=['level', 'dual']
                            )
        df.name = variable_block
        return df 
    if variable_block is not None:
        return _res_variable(res_dict, variable_block)
    else:
        res = dict()
        for variable_name in res_dict['variable_name'].keys():
            res.update({variable_name: _res_variable(res_dict, variable_name)})
        return res 


def res_constraint(res_dict, constraint_block=None):
    """
    对full_result中的constraint部分进行formation,
    返回其变量值与dual value

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
    def _res_constraint(res_dict, constraint_block):
        assert constraint_block in res_dict['constraint_name'].keys(), "不存在名为{}的constraint block".format(constraint_block)
        if isinstance(res_dict['constraint_name'][constraint_block], list):
            df = pd.DataFrame(
                {
                    'level': res_dict['constraint_level'][constraint_block],
                    'dual': res_dict['constraint_dual'][constraint_block]
                }, index=res_dict['constraint_name'][constraint_block]
            )
        else:
            df = pd.Series(
                [res_dict['constraint_level'][constraint_block][0],
                res_dict['constraint_dual'][constraint_block][0]], index=['level', 'dual']
            )
        df.name = constraint_block
        return df 
    if constraint_block is not None:
        return _res_constraint(res_dict, constraint_block)
    else:
        res = dict()
        for constraint_name in res_dict['constraint_name'].keys():
            res.update({constraint_name: _res_constraint(res_dict, constraint_name)})
        return res 




    


    
