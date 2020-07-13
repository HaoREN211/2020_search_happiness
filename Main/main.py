# @Time    : 2020/7/11 23:24
# @Author  : REN Hao
# @FileName: main.py
# @Software: PyCharm

from Main import *
from Features.XGBoost_feature_selection import *

if __name__ == '__main__':
    self = Config()

    # 查看数据缺失情况
    self.data.info(verbose=True, null_counts=True)

    # 观察数据大小
    self.data.shape

    # 简单查看数据
    self.data.head()

    # 查看label分布
    self.data["happiness"].value_counts()

    columns = select_features(self.data, ["label", "happiness"], "label")
    result = pd.DataFrame({"columns": columns})
    result.to_excel("Features/label_1_features.xlsx", index=None)






