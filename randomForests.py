# /usr/bin/python 
# --*-- coding:UTF-8 --*--
# Date 2021/10/24 11:59

import numpy as np
from collections import Counter
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class BaggingClassifier():
    def __init__(self, n_model=10):
        '''
        初始化函数
        '''
        # 分类器的数量，默认为10
        self.n_model = n_model
        # 用于保存模型的列表，训练好分类器后将对象append进去即可
        self.models = []

    def fit(self, feature=None, label=None):
        '''
        训练模型，请记得将模型保存至self.models
        :param feature: 训练集数据，类型为ndarray
        :param label: 训练集标签，类型为ndarray
        :return: None
        '''
        # ************* Begin ************#
        n = len(feature)
        for i in range(self.n_model):
            # 在训练集N随机选取n个样本  #frac=1 样本大小有放回
            randomSamples = feature.sample(n, replace=True, axis=0)
            # print(len(set(randomSamples.index.tolist()))) 大约每次选取N的2/3
            # 在所有特征M随机选取m个特征 特征无重复
            randomFeatures = randomSamples.sample(frac=1, replace=False, axis=1)
            # print(randomFeatures.columns.tolist())
            tags = self.connect(randomFeatures.columns.tolist())
            # print(tags)
            # 筛选出索引与上相同的lable
            randomLable = label.loc[randomSamples.index.tolist(),:]
            # for i,j in zip(randomFeatures.index.tolist(),randomLable.index.tolist()):
            #     print(i,j)
            model = DecisionTreeClassifier()
            model = model.fit(randomFeatures, randomLable)
            self.models.append({tags: model})
        # ************* End **************#

    def predict(self, features, target):
        '''
        :param features: 测试集数据，类型为ndarray
        :return: 预测结果，类型为ndarray
        '''
        # ************* Begin ************#
        result = []
        vote = []

        for model in self.models:
            # 获取模型的训练标签
            modelFeatures = list(model.keys())[0].split('000')[:-1]
            # print(modelFeatures)
            # 提取模型相对应标签数据
            feature = features[modelFeatures]
            # print(feature)
            # 基分类器进行预测
            r = list(model.values())[0].predict(feature)
            vote.append(r)
        # 将数组转换为矩阵 10行45列
        vote = np.array(vote)
        # print(vote.shape) # print(vote)

        for i in range(len(features)):
            # 对每棵树的投票结果进行排序选取最大的
            v = sorted(Counter(vote[:, i]).items(),
                       key=lambda x: x[1], reverse=True)
            print(v, "---",list(target)[i])
            result.append(v[0][0])
        #print(result)
        return result
        # ************* End **************#

    def dataset(self):
        iris = load_iris()
        feature = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        target = pd.DataFrame(data=map(lambda item: iris.target_names[item],
                                       iris.target), columns={'target_names'})
        # iris_datasets = pd.concat([feature, target], axis=1)
        # print(iris_datasets)
        feature_train, feature_test, target_train, target_test = \
            train_test_split(feature, target, test_size=0.3)
        # print(feature_train,target_train)
        return feature_train, feature_test, target_train, target_test

    def connect(self, ls):
        s = ''
        for i in ls:
            s += i + '000'
        return s


if __name__ == '__main__':
    Bcf = BaggingClassifier()
    featureAndTarget = Bcf.dataset()
    Bcf.fit(featureAndTarget[0],featureAndTarget[2])
    res = Bcf.predict(features=featureAndTarget[1], target=featureAndTarget[3]['target_names'])
    right = 0
    for i, j in zip(featureAndTarget[3]['target_names'], res):
        if i == j:
            right += 1
        #print(i + '\t' + j)
    print('准确率为' + str(right / len(res) * 100) + "%")
