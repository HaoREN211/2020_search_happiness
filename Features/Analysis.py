# @Time    : 2020/7/12 11:42
# @Author  : REN Hao
# @FileName: Analysis.py
# @Software: PyCharm

from Main import *
import matplotlib.pyplot as plt


# 画出各类用户该特征的取值分布
def box_plot(data, feature, label="happiness"):
    label_values = list(set(data[label].values))
    for index, value in enumerate(label_values):
        ax1 = plt.subplot(2, 3, index+1)
        sub_data = data[data[label] == value]
        ax1.set_title("label="+str(value))
        ax1.boxplot(sub_data[feature].values)
    plt.show()

