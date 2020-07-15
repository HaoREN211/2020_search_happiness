# @Time    : 2020/7/11 23:24
# @Author  : REN Hao
# @FileName: main.py
# @Software: PyCharm

from Main import *
from Features.XGBoost_feature_selection import *
from Features.function import calculate_iv

if __name__ == '__main__':
    self = Config()

    # 查看数据缺失情况
    self.data.info(verbose=True, null_counts=True)

    # 各特征与标签的皮尔森线性相关性
    for i in range(1, 6):
        self.set_label(i)
        result_corr = self.data.corr()["label"]
        pd_corr = pd.DataFrame({"column": result_corr.index,
                                "correlation": [round(x, 4) for x in result_corr.values],
                                "corr_abs": [round(abs(x), 4) for x in result_corr.values]})
        pd_corr.to_excel("corr_"+str(i)+".xlsx", index=None)

    # 各特征与标签的IV值
    for i in range(1,6):
        self.set_label(i)
        result = pd.DataFrame(columns=["column", "iv"])
        for col_index, col_value in enumerate(self.data.columns):
            if col_value == "label":
                continue
            if len(set(self.data[col_value].values))>20:
                continue
            result = result.append(pd.DataFrame({"column": [col_value],
                                                 "iv": [calculate_iv(self.data, col_value, "label")]}))
        result.to_excel("iv_"+str(i)+".xlsx", index=None)

    # 观察数据大小
    self.data.shape

    # 简单查看数据
    self.data.head()

    # 查看label分布
    self.data["happiness"].value_counts()

    columns = select_features(self.data, ["label", "happiness"], "label")
    result = pd.DataFrame({"columns": columns})
    result.to_excel("Features/label_1_features.xlsx", index=None)






