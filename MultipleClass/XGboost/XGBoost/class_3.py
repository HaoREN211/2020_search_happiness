# -*- coding: UTF-8 -*- 
# 作者：hao.ren3
# 时间：2020/7/14 19:56
# IDE：PyCharm

from Features.function import *
from Features.XGBoost_feature_selection import *
import os

DROP_COLUMNS = ["happiness", "label", "survey_time", "edu_other"]

if __name__ == '__main__':
    self = Config()
    self.set_label(3)

    # 判断调查问卷的类型与1类标签的相关性
    calculate_iv(self.data, "equity", "label")

    # 根据XGBoost选择合适的特征
    file_path = "Features/label_3_features.xlsx"
    if not os.path.exists(file_path):
        columns = select_features(self.data, ["label", "happiness"], "label")
        result = pd.DataFrame({"columns": columns})
        result.to_excel(file_path, index=None)
    else:
        columns = pd.read_excel(file_path)["columns"].values

    # 拆分训练集和验证集
    X, y = self.data[columns], self.data["label"].values
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

    # 训练模型
    model = XGBClassifier()
    model.fit(train_X, train_y)

    # 验证模型
    val_y_predict = model.predict(val_X)
    val_y_prob = model.predict_proba(val_X)
    val_0_prob, val_1_prob = [round(x[0], 4) for x in val_y_prob], [round(x[1], 4) for x in val_y_prob]
    result = pd.DataFrame({"predict": val_y_predict,
                           "val_0_prob": val_0_prob,
                           "val_1_prob": val_1_prob})
    print(accuracy_score(val_y, val_y_predict))

