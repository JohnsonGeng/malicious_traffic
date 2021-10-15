#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''

@File    :   detector.py
@Author  :   你牙上有辣子
@Contact :   johnsogunn23@gmail.com
@Create Time : 2021-10-14 16:28     
@Discription :

用于训练强化学习选定算法的检测器

'''

# import lib
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib
import time
import os


# 模型存放位置
model_path = '../result/'


# 算法池，根据算法名取对应的机器学习模型
Algorithm_POOL = {
    'RandomForest': RandomForestClassifier(random_state=0, n_estimators=50),
    'KNN': KNeighborsClassifier(),
    'NB': GaussianNB(),
    'DT': DecisionTreeClassifier(),
    'MLP': MLPClassifier(hidden_layer_sizes=(32,16), solver='adam', alpha=1e-5),
    'Ada': AdaBoostClassifier(n_estimators=100),
    'BAGGING': BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.5),
    # SVM 维度太大不太好收敛
    'SVM': SVC(kernel='rbf', probability=True, gamma='auto', max_iter=1000),
    'GBDT': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                     max_depth=1, random_state=0)
}




# 检测器类
class Detector():
    def __init__(self, algorithm):

        self.algorithm = algorithm
        self.detector = Algorithm_POOL[self.algorithm]



    # 给入数据和特征，进行检测器训练
    def train_and_test(self, data, label):

        # 返回的result字典，里面存有分类的各种评价结果
        result = {}

        # 划分训练集、测试集
        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0)

        # 训练
        train_start = time.time()
        self.detector.fit(x_train, y_train)
        train_end = time.time()
        train_time = train_end - train_start
        sample_number = len(x_test)

        # 测试
        detect_start = time.time()
        y_predict = self.detector.predict(x_test)
        detect_end = time.time()
        detect_time = detect_end - detect_start

        # 获取混淆矩阵，得到各个指标
        cm = metrics.confusion_matrix(y_test, y_predict)
        TP = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        TN = cm[1][1]

        # 准确率
        accuracy = metrics.accuracy_score(y_test, y_predict)
        # 精确率
        precision = metrics.precision_score(y_test, y_predict, pos_label='1', average='binary')
        # 召回率
        recall = metrics.recall_score(y_test, y_predict, pos_label='1', average='binary')
        # F1 Score
        f1_score = metrics.f1_score(y_test, y_predict, pos_label='1', average='binary')
        # 误警率
        false_alarm_rate = FP / (FP + TN)
        # 漏警率
        miss_alarm_rate = FN / (TP + FN)

        result['Accuracy'] = accuracy
        result['Precision'] = precision
        result['Recall'] = recall
        result['F1 Score'] = f1_score
        result['False Alarm Rate'] = false_alarm_rate
        result['Miss Alarm Rate'] = miss_alarm_rate
        result['Train Time'] = train_time
        result['Detect Time For Per Sample'] = detect_time/sample_number

        return result



    # 测试模型性能
    def test(self, data, label):

        # 返回的result字典，里面存有分类的各种评价结果
        result = {}
        # 存放提取后的特征

        detect_start = time.time()
        predict = self.detector.predict(data)
        detect_end = time.time()
        detect_time = detect_end - detect_start

        # 获取混淆矩阵，得到各个指标
        cm = metrics.confusion_matrix(label, predict)
        TP = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        TN = cm[1][1]

        # 准确率
        accuracy = metrics.accuracy_score(label, predict)
        # 精确率
        precision = metrics.precision_score(label, predict, pos_label='1', average='binary')
        # 召回率
        recall = metrics.recall_score(label, predict, pos_label='1', average='binary')
        # F1 Score
        f1_score = metrics.f1_score(label, predict, pos_label='1', average='binary')
        # 误警率
        false_alarm_rate = FP / (FP + TN)
        # 漏警率
        miss_alarm_rate = FN / (TP + FN)


        result['Accuracy'] = accuracy
        result['Precision'] = precision
        result['Recall'] = recall
        result['F1 Score'] = f1_score
        result['False Alarm Rate'] = false_alarm_rate
        result['Miss Alarm Rate'] = miss_alarm_rate
        result['Detect Time For Per Sample'] = detect_time / len(label)

        return result


    # 保存模型
    def save_model(self):
        # 如果存在同样的模型
        if os.path.exists(model_path+self.algorithm+'.model'):
            os.remove(model_path+self.algorithm+'.model')

        joblib.dump(self.detector, model_path+self.algorithm+'.model')


    # 载入模型
    def load_model(self):
        self.detector = joblib.load(model_path+self.algorithm+'.model')

