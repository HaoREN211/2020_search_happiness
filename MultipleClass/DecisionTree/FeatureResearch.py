# -*- coding: UTF-8 -*- 
# 作者：hao.ren3
# 时间：2020/7/15 18:25
# IDE：PyCharm

from Main import *
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
import matplotlib.pyplot as plt
import pydotplus

# https://www2.graphviz.org/Packages/development/windows/10/msbuild/Release/Win32/

if __name__ == '__main__':
    self = Config()
    dtc = DecisionTreeClassifier()

    X, y = self.data.drop(columns=["label", "id", "happiness"]).copy(), self.data["label"].values
    dtc.fit(X, y)

    # 使用sklearn自带的包打印决策树的结构
    plt.figure()
    plot_tree(dtc, filled=True,
              feature_names=X.columns)
    plt.show()

    with open('treeone.dot', 'w') as f:
        dot_data = export_graphviz(dtc, out_file=None)
        f.write(dot_data)

    dot_data = export_graphviz(dtc, feature_names=X.columns,
                                    filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("treetwo.pdf")
