# -*- coding: UTF-8 -*- 
# 作者：hao.ren3
# 时间：2020/7/15 15:11
# IDE：PyCharm

import lightgbm
from Features.function import *
from sklearn.model_selection import KFold

# https://www.cnblogs.com/dudumiaomiao/p/9693391.html

if __name__ == '__main__':
    train_data = Config()
    test_data = Config(is_train=False)

    # n_splits 表示划分为几块（至少是2）
    # shuffle 表示是否打乱划分，默认False，即不打乱
    # random_state 表示是否固定随机起点，Used when shuffle == True.
    kf = KFold(n_splits=5, shuffle=True)
    for train, test in kf.split(train_data.data):
        print(str(len(train))+"--->"+str(len(test)))
        print(train)


    columns = list(set(train_data.data.columns).intersection(set(test_data.data.columns)))

    train_x, train_y = train_data.data[columns].drop(columns=["id", "happiness", "label"]), train_data.data["label"].values
    test_x, test_y = test_data.data[columns].drop(columns=["id", "happiness", "label"]), test_data.data["label"].values

    clf = lightgbm
    train_matrix = clf.Dataset(train_x, label=train_y)
    test_matrix = clf.Dataset(test_x, label=test_y)



    params = {
        #                 'boosting_type': 'gbdt',
        #                 'learning_rate': 0.01,
        #                 'objective': 'binary',
        #                 'metric': 'auc',
        #                 'min_child_weight': 1.5,
        #                 'num_leaves': 2 ** 5,
        #                 'lambda_l2': 10,
        #                 'subsample': 0.9,
        #                 'colsample_bytree': 0.7,
        #                 'colsample_bylevel': 0.7,
        #                 'learning_rate': 0.01,
        #                 'seed': 2017,
        #                 'nthread': 12,
        #                 'silent': True,
        'task': 'train',
        'learning_rate': 0.005,
        #                         'max_depth': 8,
        #                         'num_leaves':2**6-1,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        #                         'is_unbalance':True,
        'feature_fraction': 0.8,
        'metric': 'auc',
        'bagging_fraction': 0.86,

        #                         'lambda_l1': 0.0001,
        'lambda_l2': 49,
        'bagging_freq': 3,
        #                         'min_data_in_leaf':5,
        'verbose': 1,
        'random_state': 2267,
    }

    num_round = 10000
    early_stopping_rounds = 300
    model = clf.train(params, train_matrix, num_round, valid_sets=test_matrix,
                      early_stopping_rounds=early_stopping_rounds, verbose_eval=300
                      )
    model.predict(test_x)
