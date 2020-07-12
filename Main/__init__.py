# @Time    : 2020/7/11 23:23
# @Author  : REN Hao
# @FileName: __init__.py.py
# @Software: PyCharm

import numpy as np
import pandas as pd


class Config(object):
    TRAIN_DATA_FILE = r"Data/happiness_train_complete.csv"
    TEST_DATA_FILE = r"Data/happiness_test_complete.csv"

    def __init__(self, is_train=True):
        tmp_file = self.TRAIN_DATA_FILE if is_train else self.TEST_DATA_FILE
        tmp_type = "训练" if is_train else "测试"
        self.data = pd.read_csv(tmp_file, header=0, encoding='gbk')
        print("载入" + tmp_type + "数据集，覆盖"
              + str(np.shape(self.data)[0]) + "样本，"
              + str(np.shape(self.data)[1]) + "特征")
        if is_train:
            self.drop_invalid_data_label()
        self.calculate_age()

    # 多分类标签时用
    def set_label(self, label=1):
        print("当前处理"+str(label)+"类标签")
        self.data["label"] = self.data.apply(lambda x: 1 if x["happiness"] == label else 0, axis=1)

    # 删除标签不正确的样本
    def drop_invalid_data_label(self):
        print("删除标签不正确的样本")
        self.data = self.data[self.data["happiness"] > 0]

    # 根据用户的出生年月日计算当前岁数
    def calculate_age(self, current_year=2020):
        print("正在计算用户的年龄")
        self.data["age"] = self.data[["birth"]].apply(lambda x: int(current_year-int(x["birth"])), axis=1)
