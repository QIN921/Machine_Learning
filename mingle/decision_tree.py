import pandas as pd
from pandas import DataFrame
import numpy as np


# 计算信息熵
def cal_entropy(data: DataFrame):
    data_label = data.iloc[:, -1]
    label_class = data_label.value_counts()
    entropy = 0
    for item in label_class.keys():
        p_k = label_class[item] / len(data_label)
        entropy += - p_k * np.log2(p_k)
    return entropy


# 计算给定属性的信息增益
def cal_entropy_gain(data: DataFrame, item: str):
    entropy = cal_entropy(data)
    feature_class = data[item].value_counts()
    gain = entropy
    for i in feature_class.keys():
        weight = feature_class[i] / data.shape[0]
        # 计算某一固定特征分类后的熵
        ent_i = cal_entropy(data.loc[data[item] == i])
        gain -= weight * ent_i
    return gain


def get_best_feature(data: DataFrame):
    features = data.columns[:-1]
    # print(features)
    res = {}
    for item in features:
        temp = cal_entropy_gain(data, item)
        res[item] = temp
    # res.items()把字典转换为了元组的列表
    res = sorted(res.items(), key=lambda x: x[1], reverse=True)
    # res = [('纹理', 0.3805918973682686), ('脐部', 0.289158782841679), ...]
    return res[0][0]


# 获取标签最多的那一类
def get_most_label(data):
    data_label = data.iloc[:, -1]
    label_sort = data_label.value_counts(sort=True)
    return label_sort.keys()[0]


# 将数据转化为（属性值：数据）的元组形式返回，并删除之前的特征列
def drop_exist_feature(data, best_feature):
    attr = pd.unique(data[best_feature])
    new_data = [(nd, data[data[best_feature] == nd]) for nd in attr]
    new_data = [(n[0], n[1].drop([best_feature], axis=1)) for n in new_data]
    return new_data


# 创建决策树
def create_tree(data: DataFrame):
    # 获取最后一列, 即分类结果
    data_label = data.iloc[:, -1]
    # print(data_label)
    if len(data_label.value_counts()) == 1:
        # 只有一种特征
        return data_label.values[0]
    if all(len(data[i].value_counts()) == 1 for i in data.iloc[:, :-1].columns):
        # 所有数据特征值一样, 选样本最多的类作为分类
        return get_most_label(data)
    best_feature = get_best_feature(data)
    tree = {best_feature: {}}
    exist_vals = pd.unique(data[best_feature])  # 当前数据下最佳特征的取值

    # 统计每个特征的取值情况作为全局变量
    # 输出结果: {'色泽': ['青绿', '乌黑', '浅白'], '根蒂': ['蜷缩', '稍蜷', '硬挺'], ...}
    column_count = dict([(ds, list(pd.unique(data[ds]))) for ds in data.iloc[:, :-1].columns])

    if len(exist_vals) != len(column_count[best_feature]):  # 如果特征的取值相比于原来的少了
        no_exist_attr = set(column_count[best_feature]) - set(exist_vals)  # 少的那些特征
        for no_feat in no_exist_attr:
            tree[best_feature][no_feat] = get_most_label(data)  # 缺失的特征分类为当前类别最多的

    for item in drop_exist_feature(data, best_feature):  # 根据特征值的不同递归创建决策树
        tree[best_feature][item[0]] = create_tree(item[1])
    return tree


def main():
    data: DataFrame = pd.read_csv('./data/data_word.csv')
    # print(data)

    tree = create_tree(data)
    print(tree)


if __name__ == '__main__':
    main()
