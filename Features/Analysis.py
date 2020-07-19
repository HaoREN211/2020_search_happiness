# @Time    : 2020/7/12 11:42
# @Author  : REN Hao
# @FileName: Analysis.py
# @Software: PyCharm

from Main import *
from matplotlib import pyplot as plt, font_manager
import os


# 画出各类用户该特征的取值分布
def box_plot(data, feature, label="happiness"):
    label_values = list(set(data[label].values))
    for index, value in enumerate(label_values):
        ax1 = plt.subplot(2, 3, index + 1)
        sub_data = data[data[label] == value]
        ax1.set_title("label=" + str(value))
        ax1.boxplot(sub_data[feature].values)
    plt.show()


# 画出label=0和=1两种情况下，特征的占比分布情况
def line_plot_percentage(data, feature, label="label"):
    list_value = list(set([int(x) for x in data[feature].values]))
    label_0, label_1 = data[data[label] == 0][feature].values, data[data[label] == 1][feature].values
    result = pd.DataFrame(columns=["value", "label_0_ratio", "label_1_ratio"])
    for index, current_value in enumerate(list_value):
        result = result.append(
            pd.DataFrame({
                "value": [current_value],
                "label_0_ratio": [sum([1 if x == current_value else 0 for x in label_0]) / len(label_0)],
                "label_1_ratio": [sum([1 if x == current_value else 0 for x in label_1]) / len(label_1)],
            })
        )
    result.sort_values(by=["value"], inplace=True)

    my_font = font_manager.FontProperties(fname="C:\Windows\Fonts\simhei.ttf")
    plt.plot(result["value"].values, result["label_0_ratio"].values, color='b', label='label_0')
    plt.plot(result["value"].values, result["label_1_ratio"].values, color='r', label='label_1')
    plt.legend()
    plt.title("各类用户在" + feature + "特征下的分布情况",
              fontproperties=my_font)
    # file_path = r"./Pics/percentage/label_1/" + feature + ".png"
    # if not os.path.exists(os.path.dirname(file_path)):
    #     os.makedirs(os.path.dirname(file_path))
    # plt.savefig(file_path)
    plt.show()
    return result


# 画出特征各值下的label=0和=1两种用户占比情况
def line_plot_feature(data, feature, label="label"):
    list_value = list(set([int(x) for x in data[feature].values]))
    label_0, label_1 = data[data[label] == 0][feature].values, data[data[label] == 1][feature].values
    result = pd.DataFrame(columns=["value", "label_0_ratio", "label_1_ratio"])
    for index, current_value in enumerate(list_value):
        label_0_cnt = sum([1 if x == current_value else 0 for x in label_0])
        label_1_cnt = sum([1 if x == current_value else 0 for x in label_1])
        total_cnt = label_0_cnt + label_1_cnt
        result = result.append(
            pd.DataFrame({
                "value": [current_value],
                "label_0_ratio": [label_0_cnt / total_cnt],
                "label_1_ratio": [label_1_cnt / total_cnt]
            })
        )
    result.sort_values(by=["value"], inplace=True)

    plt.plot(result["value"].values, result["label_0_ratio"].values, color='b', label='label_0')
    plt.plot(result["value"].values, result["label_1_ratio"].values, color='r', label='label_1')
    plt.show()
    return result
