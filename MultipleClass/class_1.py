# @Time    : 2020/7/12 17:53
# @Author  : REN Hao
# @FileName: class_1.py
# @Software: PyCharm

from Main import *
from Features.function import *
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

if __name__ == '__main__':
    self = Config()
    self.set_label(1)

    # 判断调查问卷的类型与1类标签的相关性
    calculate_iv(self.data, "survey_type", "label")

    result = pd.DataFrame(columns=["column", "iv"])
    for index, column in enumerate(self.data.columns):
        if column in ("happiness", "label"):
            continue
        if len(list(set(self.data[column].values))) < 10:
            result = result.append(
                pd.DataFrame({
                    "column": [column],
                    "iv": [calculate_iv(self.data, column, "label")]
                })
            )
            print(column+"的iv值为："+str(calculate_iv(self.data, column, "label")))
    np.median(self.data["family_income"].median)

    # 将地区的编码转化成为OneHot编码
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_df = pd.DataFrame(one_hot_encoder.fit_transform(
        np.reshape(self.data["county"].values, (-1, 1))))
