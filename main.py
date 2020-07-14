# -*- coding: UTF-8 -*- 
# 作者：hao.ren3
# 时间：2020/7/14 19:59
# IDE：PyCharm

from Features.function import *
from Features.XGBoost_feature_selection import *
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    data_train = Config(is_train=True)
    data_test = Config(is_train=False)

    val_result = pd.DataFrame(columns=["id", "class_1",  "class_2",  "class_3",  "class_4",  "class_5", "happiness"])
    test_result = pd.DataFrame(columns=["id", "class_1", "class_2", "class_3", "class_4", "class_5", "happiness"])

    for current_class in range(1,6):
        print("当前正在计算幸福指数为1的模型")
        data_train.set_label(current_class)

        # 读取当前标签使用的特征列表
        columns = list(pd.read_excel("Features/label_"+str(current_class)+"_features.xlsx")["columns"].values)
        columns.append("id")
        columns.append("happiness")
        columns = list(filter(lambda x: x in data_test.data.columns, columns))
        print("--- 当前使用"+str(len(columns))+"个特征训练模型")

        print("--- 正在加工训练数据集和测试数据集")
        X, y = data_train.data[columns], data_train.data["label"].values
        train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
        test_X = data_test.data[columns]

        if current_class == 1:
            val_result["id"] = val_X["id"].values
            val_result["happiness"] = val_X["happiness"].values
            test_result["id"] = test_X["id"].values
        train_X = train_X.drop(columns=["id", "happiness"])
        val_X = val_X.drop(columns=["id", "happiness"])
        test_X = test_X.drop(columns=["id", "happiness"])

        # 训练模型
        print("--- 正在训练模型")
        model = XGBClassifier()
        model.fit(train_X, train_y)

        # 预测数据
        print("--- 正在预测数据")
        val_result["class_"+str(current_class)] = [round(x[1], 3) for x in model.predict_proba(val_X)]
        test_result["class_" + str(current_class)] = [round(x[1],3) for x in model.predict_proba(test_X)]

    # 逻辑回归模型
    log_model = LogisticRegression()

    # 训练逻辑回归模型
    log_model.fit(val_result[["class_1",  "class_2",  "class_3",  "class_4",  "class_5"]], val_result["happiness"].values)

    # 预测y的值
    test_result["happiness"] = log_model.predict(test_result[["class_1",  "class_2",  "class_3",  "class_4",  "class_5"]])

    test_result["happiness"].value_counts()

    test_result[["id", "happiness"]].to_csv("result.csv", index=False)
