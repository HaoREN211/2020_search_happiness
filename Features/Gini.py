# @Time    : 2020/7/21 22:25
# @Author  : REN Hao
# @FileName: Gini.py
# @Software: PyCharm
from copy import copy
from Main import *


# 计算离散特征column的gini值
def calculate_gini(data, column, label="label"):
    final_gini = 0
    for column_value in list(set(data[column].values)):
        sub_data = data[data[column] == column_value].copy()
        possibility = float(len(sub_data))/float(len(data))
        current_gini = 1
        for label_value in list(set(sub_data[label].values)):
            current_gini -= (float(len(sub_data[sub_data[label] == label_value]))/float(len(sub_data)))**2
        final_gini += possibility * current_gini
    return final_gini


# 计算基尼系数
def calculate_gini_by_interval(data, column_name, column_min, column_max=None, label="label"):
    column_max = copy(column_min) if column_max is None else column_max
    result = pd.DataFrame({
        "column": data.apply(lambda x: 1 if column_min <= x[column_name] <= column_max else 0, axis=1),
        "label": data[label].values
    })
    return calculate_gini(result, "column", "label")
