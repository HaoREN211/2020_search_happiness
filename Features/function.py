# @Time    : 2020/7/12 22:22
# @Author  : REN Hao
# @FileName: function.py
# @Software: PyCharm
from Main import *
import math


# 计算离散变量column的iv值
def calculate_iv(data, column, label):
    result_iv = 0
    nb_good, nb_bad = calculate_good_bad(data, label)
    temp_result = calculate_woe_detail(data, column, label)
    for current_row in temp_result.itertuples():
        current_value, current_woe = getattr(current_row, "value"), getattr(current_row, "woe")
        current_data = data[data[column] == current_value][[column, label]]
        current_good, current_bad = calculate_good_bad(current_data, label)
        result_iv += ((current_bad/nb_bad)-(current_good/nb_good)) * current_woe
    return round(result_iv, 3)


# 计算好坏样本在data数据集中的样本数
def calculate_good_bad(data, label):
    return len(data[data[label] == 0]), len(data[data[label] == 1])


def calculate_woe_detail(data, column, label):
    nb_good, nb_bad = calculate_good_bad(data, label)
    list_value = set(data[column].values)
    final_result = pd.DataFrame(columns=["value", "woe"])
    for current_value in list_value:
        current_data = data[data[column] == current_value][[column, label]]
        current_good, current_bad = calculate_good_bad(current_data, label)
        if current_good == 0 and current_bad == 0:
            result = 0
        elif current_bad == 0:
            result = -100
        elif current_good == 0:
            result = 100
        else:
            result = (current_bad/nb_bad)/(current_good/nb_good)
            result = math.log(result, math.e)
        final_result = final_result.append(pd.DataFrame({"value": [current_value],
                                                        "woe": [result]}))
    return final_result
