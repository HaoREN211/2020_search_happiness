# @Time    : 2020/7/12 22:22
# @Author  : REN Hao
# @FileName: function.py
# @Software: PyCharm
from Main import *
import math


def find_2_column_with_max_id(data, label="label", value_cnt=20):
    # 统计数据各特征有多少属性值
    column_value_cnt = [len(set(data[x].values)) for x in data.columns]

    drop_column = ["happiness", "label"]

    # 只保留取值个数小于等于value_cnt的离散值
    column_filtered = data.columns[[True if x <= value_cnt else False for x in column_value_cnt]]

    result = pd.DataFrame(columns=["column_1", "column_1_min", "column_1_max",
                                   "column_2", "column_2_min", "column_2_max", "iv"])

    # 寻找两两组合的特征
    for index_1, column_1 in enumerate(column_filtered):
        list_value_1 = sorted(list(set(data[column_1].values)))
        if column_1 in drop_column:
            continue
        for index_2, column_2 in enumerate(column_filtered):
            if index_2 <= index_1:
                continue
            if column_2 in drop_column:
                continue
            list_value_2 = sorted(list(set(data[column_2].values)))
            # 寻找两个特征合适的区间
            print("正在计算复合指标"+str(column_1)+"和"+str(column_2))
            for index_1_min, column_1_min in enumerate(list_value_1):
                for column_1_max in list_value_1[index_1_min+1:]:
                    for index_2_min, column_2_min in enumerate(list_value_2):
                        for column_2_max in list_value_2[index_2_min+1:]:
                            data["complex_feature"] = data.apply(
                                lambda x: 1 if ((column_1_min <= x[column_1] <= column_1_max)
                                                and (column_2_min <= x[column_2] <= column_2_max))
                                else 0, axis=1)
                            result = result.append(pd.DataFrame({
                                "column_1": [column_1],
                                "column_1_min": [column_1_min],
                                "column_1_max": [column_1_max],
                                "column_2": [column_2],
                                "column_2_min": [column_2_min],
                                "column_2_max": [column_2_max],
                                "iv": [calculate_iv(data, "complex_feature", label=label)]
                            }))
    return result


# 计算离散变量column的iv值
def calculate_iv(data, column, label):
    result_iv = 0
    nb_good, nb_bad = calculate_good_bad(data, label)
    temp_result = calculate_woe_detail(data, column, label)
    for current_row in temp_result.itertuples():
        current_value, current_woe = getattr(current_row, "value"), getattr(current_row, "woe")
        current_data = data[data[column] == current_value][[column, label]]
        current_good, current_bad = calculate_good_bad(current_data, label)
        result_iv += ((current_bad / nb_bad) - (current_good / nb_good)) * current_woe
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
            result = -1
        elif current_good == 0:
            result = 1
        else:
            result = (current_bad / nb_bad) / (current_good / nb_good)
            result = math.log(result, math.e)
        final_result = final_result.append(pd.DataFrame({"value": [current_value],
                                                         "woe": [result]}))
    return final_result
