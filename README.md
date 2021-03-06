#### 数据集

    采用 NSL-KDD 数据集，存放于 data/detection 文件夹下，以CSV文件格式组织
    
包含以下四个文件：

    KDDTrain+.csv ———— 训练数据集，一般用这个训练
    KDDTrain+_20Percent.csv ———— 20%的训练数据集
    KDDTest+.csv ———— 测试数据集，一般用这个测试
    KDDTest-21.csv ———— 难度更高的测试数据集


#### 文件结构（Source文件夹中）

    train：用于直接训练模型
    utils：工具类，执行数据读取、数据预处理与画图操作
    start：强化学习训练智能体筛选模型的入口
    detector：检测器类，实现了检测器的训练、测试、保存与载入
    env：环境，用于智能体获取回报
    agent：智能体类，用于构建DQN
    action_value：获取奖励值
    
    
#### 输入输出

    输入：数据和标签
        
        通过 utils.DataLoader(特征列表) 获取
        
        DataLoader.data 提取特征列表中指定特征后的训练数据
        DataLoader.label 标签
    
    输出：模型对流量的检测结果字典
        
        通过 detector.Detector('学习算法') 获取
        
        Detector.train_and_test(数据, 标签) 训练并测试
        Detector.test(数据, 标签) 评估模型

    
#### 最终经强化学习搜索得到性能最佳的模型

    随机森林算法 + 9个特征（1, 2, 5, 22, 24, 25, 27, 35）



#### 搜索出的其他机器学习算法性能较好的模型，供参考

    决策树 + 10个特征（39, 15, 4, 1, 21, 31, 34, 11, 37, 38）
    KNN + 10个特征（5, 24, 31, 2, 20, 4, 25, 11, 3, 22）
    Bagging +  8个特征（2, 29, 31, 22, 3, 4, 1, 12） 
    GBDT + 9个特征（4, 3, 8, 1, 2, 26, 9, 11, 33）