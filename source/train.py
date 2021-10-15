#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''

@File    :   train.py
@Author  :   你牙上有辣子
@Contact :   johnsogunn23@gmail.com
@Create Time : 2021-10-14 16:23     
@Discription :

主函数

'''

# import lib
from detector import Detector
from utils import *

# 由强化学习智能体选择 NSL-KDD 中的第1，2，5，22，24，25，27，35，36号特征
RF_feature = [1, 2, 5, 22, 24, 25, 27, 35]
DT_feature = [2, 4, 5, 24, 27, 30, 31, 33, 35, 36]

if __name__ == '__main__':

    # 根据特征获取数据
    dl = DataLoader(feature=RF_feature)
    data, label = dl.load_data('KDDTrain+.csv')
    # 对数据做min-max变换
    data = data_preprocessing(data)
    # 使用随机森里训练模型
    detector = Detector('RandomForest')
    # 获得模型结果
    result = detector.train_and_test(data, label)
    print('训练结果：')
    print(result)
    # 保存模型
    detector.save_model()

    # 用KDDTest+数据集评估性能
    dl_evaluate = DataLoader(feature=RF_feature)
    test_data, test_label = dl.load_data('KDDTest-21.csv')
    test_data = data_preprocessing(test_data)
    eva_result = detector.test(test_data, test_label)
    print('测试结果：')
    print(eva_result)
