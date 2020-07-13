# @Time    : 2020/7/11 23:23
# @Author  : REN Hao
# @FileName: __init__.py.py
# @Software: PyCharm

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


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
        self.generate_feature()
        self.replace_default_value()
        self.generate_replaced_feature()
        self.one_hot_feature()
        self.drop_non_useful_columns()



    # 多分类标签时用
    def set_label(self, label=1):
        print("当前处理"+str(label)+"类标签")
        self.data["label"] = self.data.apply(lambda x: 1 if x["happiness"] == label else 0, axis=1)

    # 和伴侣文化程度的差异
    def edu_diff(self):
        self.data["edu_diff"] = self.data.apply(lambda x: x["edu"]-x["s_edu"] if x["s_edu"] > 0 else 0, axis=1)

    # 删除标签不正确的样本
    def drop_invalid_data_label(self):
        print("删除标签不正确的样本")
        self.data = self.data[self.data["happiness"] > 0]

    # 根据用户的出生年月日计算当前岁数
    def calculate_age(self, current_year=2020):
        print("正在计算用户的年龄")
        self.data["age"] = self.data[["birth"]].apply(lambda x: int(current_year-int(x["birth"])), axis=1)

    # 计算毕业至今的年数
    def calculate_edu_td_yr(self, current_year=2020):
        print("计算毕业至今的年数")
        self.data["edu_td_yr"] = self.data.apply(lambda x: int(current_year-x["edu_yr"]) if x["edu_yr"]>0 else 0, axis=1)

    # 计算毕业时的年纪
    def calculate_edu_age(self):
        print("计算毕业时的年纪")
        self.data["edu_age"] = self.data.apply(lambda x: int(x["edu_yr"] - x["birth"]) if x["edu_yr"] > 0 else 0, axis=1)

    # 用户10年前后地位的变化
    def class_diff(self):
        print("计算用户10年前后地位的变化情况")
        self.data["class_10_before_diff"] = self.data.apply(lambda x: x["class"]-x["class_10_before"], axis=1)
        self.data["class_10_after_diff"] = self.data.apply(lambda x: x["class_10_after"] - x["class"], axis=1)

    # 样本收入相关的特征
    def income_about(self):
        print("样本收入相关的特征")
        # 个人收入占家庭总收入
        self.data["income_percentage"] = self.data.apply(lambda x: x["income"]/x["family_income"] if x["family_income"]>0 else 0, axis=1)

        # 家庭成员的平均收入
        self.data["income_per_person"] = self.data.apply(
            lambda x: x["family_income"] / x["family_m"] if x["family_m"] > 0 else x["family_income"], axis=1)


    # 替换缺失值之后计算新的特征
    def generate_replaced_feature(self):
        print("替换缺失值之后计算新的特征")
        self.class_diff()
        self.income_about()
        self.calculate_marital_age()

    def calculate_marital_age(self):
        self.data["marital_age"] = self.data.apply(lambda x: x["marital_1st"]-x["birth"] if x["marital_1st"]>1000 else 0, axis=1)
        self.data["s_age_diff"] = self.data.apply(lambda x: x["s_birth"] - x["birth"] if x["s_birth"] > 1000 else 0, axis=1)
        self.data["second_marital"] = self.data.apply(lambda x: 1 if (x["marital_1st"] == x["marital_now"])
                                                                     and (x["marital_now"] > 1000) else 0, axis=1)

    # 替换缺失值之前计算新的特征
    def generate_feature(self):
        print("替换缺失值之前计算新的特征")
        self.calculate_age()
        self.calculate_edu_td_yr()
        self.calculate_edu_age()
        self.edu_diff()

    # 替换特征的异常值或缺失值
    def replace_default_value(self):
        print("正在替换异常值和缺失值：")
        columns = ["nationality", "religion", "edu", "edu_status", "income", "political", "health", "health_problem",
                   "depression", "hukou_loc", "media_1", "media_2", "media_3", "media_4", "media_5", "media_6",
                   "leisure_1", "leisure_2", "leisure_3", "leisure_4", "leisure_5", "leisure_6", "leisure_7", "leisure_8",
                   "leisure_9", "leisure_10", "leisure_11", "leisure_12", "socialize", "relax", "learn", "social_neighbor",
                   "social_friend", "socia_outing", "equity", "class", "class_10_before", "class_10_after", "class_14",
                   "insur_1", "insur_2", "insur_3", "insur_4", "family_income", "family_m", "family_status", "house",
                   "car", "son", "daughter", "minor_child", "s_edu"]
        for column in columns:
            print("---> 当前替换"+column+", 替换成众数："+str(self.data[column].median()))
            self.data[column] = self.data.apply(lambda x: self.data[column].median() if x[column]<0 else x[column], axis=1)
            self.data[column] = self.data[column].fillna(self.data[column].median())

        columns = ["work_status", "work_yr", "work_type", "work_manage"]
        for column in columns:
            print("---> 当前替换" + column + ", 替换成0")
            self.data[column] = self.data[column].fillna(0)
            self.data[column] = self.data.apply(lambda x: x[column] if x[column]>0 else 0, axis=1)


    # 丢弃不需要的特征
    def drop_non_useful_columns(self):
        print("丢弃不需要的特征")
        columns = ["edu_other", "edu_yr", "birth", "join_party", "property_other", "invest_other", "invest_6",
                   "marital_1st", "s_birth", "marital_now"]
        for column in columns:
            if column in self.data.columns:
                self.data = self.data.drop(columns=[column])

    # 独热类型的特征
    def one_hot_feature(self):
        print("正在处理独热类型的特征")
        one_hot_encoder = OneHotEncoder(sparse=False)
        columns = ["hukou", "hukou_loc", "work_exper", "work_status", "work_type", "work_manage", "marital"]
        for column in columns:
            list_values = list(set(self.data[column].values))
            one_hot_encoder.fit(np.reshape(self.data[column].values, (-1, 1)))
            result = one_hot_encoder.transform(np.reshape(self.data[column].values, (-1, 1))).tolist()
            for index, value in enumerate(list_values):
                category_index = list(one_hot_encoder.categories_[0]).index(value)
                current_values = [int(x[category_index]) for x in result]
                self.data[column+"_"+str(value)] = current_values
