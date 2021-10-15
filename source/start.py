#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''

@File    :   start.py
@Author  :   你牙上有辣子
@Contact :   johnsogunn23@gmail.com
@Create Time : 2021-10-15 15:11     
@Discription :

强化学习智能体训练主函数

'''

# import lib
import argparse
import time
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import optimizers
from chainerrl import explorers
import action_value as ActionValue
from agent import MyDoubleDQN
from chainerrl.replay_buffers import prioritized
import utils
from detector import Detector
import env as Env


# 命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--maxf', type=int, default=10)
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--cls', type=str, default='RandomForest')
args = parser.parse_args()


# 超参数
minibatch_size = 8
replay_start_size = 20
update_interval = 5
state_size = 10  # 可观察的状态数
action_size = 41  # 可选的特征数
feature_max_count = args.maxf  # 选取的特征数目大于该值时，reward为0，用于当特征数目在该范围内时，成功率最多可以到达多少
MAX_EPISODE = 1000 #
net_layers = [32, 64] # 一个包含两个隐藏层的神经网络

result_file = './result/result-{}-{}-{}.txt'.format(args.cls, time.strftime('%Y%m%d%H%M'),args.maxf)

# 每一轮逻辑如下
# 1. 初始化环境，定义S和A两个list，用来保存过程中的state和action。进入循环，直到当前这一轮完成（done == True）
# 2. 在每一步里，首先选择一个action，此处先用简单的act()代替
# 3. 接着env接收这个action，返回新的state，done和reward，当done==False时，reward=0，当done==True时，reward为模型的准确率
# 4. 如果done==True，那么应该把当前的S、A和reward送到replay buffer里（replay也应该在此时进行），往replay buffer里添加时，
#    每一对state和action都有一个reward，这个reward应该和env返回的reward（也就是该模型的acc）和count有关。

# 用这个逻辑替代原来的my_train的逻辑，只需要把agent加入即可，agent应该是不需要修改的



def main():

    # 保存训练中每一轮的回报
    train_reward = []
    # 保存评估中每一轮的回报
    evaluate_reward = []
    # 保存训练中每一轮的分类准确率
    train_accuracy = []
    # 保存评估中每一轮的分类准确率
    evaluate_accuracy = []
    # 用来保存效果最优的特征集合
    feature_list_train = []
    feature_list_evaluate = []

    train_precision = []
    train_recall = []



    class QFunction(chainer.Chain):
        def __init__(self, obs_size, n_actions, n_hidden_channels=None):
            super(QFunction, self).__init__()
            if n_hidden_channels is None:
                n_hidden_channels = net_layers
            net = []
            inpdim = obs_size
            for i, n_hid in enumerate(n_hidden_channels):
                net += [('l{}'.format(i), L.Linear(inpdim, n_hid))]
                net += [('norm{}'.format(i), L.BatchNormalization(n_hid))]
                net += [('_act{}'.format(i), F.relu)]
                net += [('_dropout{}'.format(i), F.dropout)]
                inpdim = n_hid

            net += [('output', L.Linear(inpdim, n_actions))]

            with self.init_scope():
                for n in net:
                    if not n[0].startswith('_'):
                        setattr(self, n[0], n[1])

            self.forward = net

        def __call__(self, x, test=False):
            """
            Args:
                x (ndarray or chainer.Variable): An observation
                test (bool): a flag indicating whether it is in test mode
            """
            for n, f in self.forward:
                if not n.startswith('_'):
                    x = getattr(self, n)(x)
                elif n.startswith('_dropout'):
                    x = f(x, 0.1)
                else:
                    x = f(x)

            return ActionValue.DiscreteActionValue(x)

    def evaluate(env, agent, current):
        for i in range(1):
            print("evaluate episode: {}".format(current))
            state = env.reset()
            terminal = False
            action_list = []

            while not terminal:
                action = agent.act(state, action_list)
                if action not in action_list:
                    action_list.append(action)
                state, reward, classify_result, terminal = env.step(action)

                if terminal or len(action_list) > 10:
                    if len(action_list) > 10:
                        terminal = True

                    with open(result_file, 'a+') as f:
                        f.write(
                            "--------------------------------------------------------------------------------------------------\n"
                            "Evaluate episode:{}, Reward = {}, Accuracy = {}, FAR = {}, MAR = {}, Action = {}\n"
                            "-------------------------------------------------------------------------------------------------\n"
                                .format(current, reward, classify_result['Accuracy'], classify_result['False Alarm Rate'], classify_result['Miss Alarm Rate'], action_list)
                        )
                        print(
                            "--------------------------------------------------------------------------------------------------\n"
                            "Evaluate episode:{}, Reward = {}, Accuracy = {}, FAR = {}, MAR = {}, Action = {}\n"
                            "-------------------------------------------------------------------------------------------------\n"
                                .format(current, reward, classify_result['Accuracy'], classify_result['False Alarm Rate'], classify_result['Miss Alarm Rate'], action_list)
                        )
                    # 加入本轮次评估的回报,后面可能需要
                    evaluate_reward.append(reward)
                    # 每次评估添加本次评估准确率
                    evaluate_accuracy.append(classify_result['Accuracy'])
                    # 同时添加对应特征
                    feature_list_evaluate.append(action_list)



    def train_agent(env, agent):
        for episode in range(MAX_EPISODE):
            state = env.reset()
            terminal = False
            reward = 0
            t = 0
            action_list = []
            while not terminal:
                t += 1
                action = agent.act_and_train(
                    state, reward, action_list)  # 此处action是否合法（即不能重复选取同一个指标）由agent判断。env默认得到的action合法。
                if action not in action_list:
                    action_list.append(action)
                state, reward, classify_result, terminal = env.step(action)
                print("Episode:{}, t:{}, Action:{}, Accuracy = {}, FAR = {}, MAR = {}, Reward = {}".format(episode, t, action_list, classify_result['Accuracy'], classify_result['False Alarm Rate'], classify_result['Miss Alarm Rate'], reward))

                if terminal:
                    with open(result_file, 'a+') as f:
                        f.write("Train episode:{}, Reward = {}, Accuracy = {}, FAR = {}, MAR = {}, Action = {}\n"
                                .format(episode, reward , classify_result['Accuracy'], classify_result['False Alarm Rate'], classify_result['Miss Alarm Rate'], action_list))
                        print("Train episode:{}, Reward = {}, Accuracy = {} ,FAR = {}, MAR = {}, Action = {}\n"
                                .format(episode, reward, classify_result['Accuracy'], classify_result['False Alarm Rate'], classify_result['Miss Alarm Rate'], action_list))

                        agent.stop_episode()
                        # 加入轮次训练的回报,后面可能需要
                        train_reward.append(reward)
                        # 加入本轮次训练的准确率
                        train_accuracy.append(classify_result['Accuracy'])
                        train_precision.append(classify_result['Precision'])
                        train_recall.append(classify_result['Recall'])
                        # 本轮次对应所选的特征
                        feature_list_train.append(action_list)
                        if (episode + 1) % 10 == 0 and episode != 0:
                            evaluate(env, agent, (episode + 1) / 10)

    def create_agent(env):
        state_size = env.state_size
        action_size = env.action_size
        q_func = QFunction(state_size, action_size)

        start_epsilon = 1
        end_epsilon = 0.3
        decay_steps = 10
        explorer = explorers.LinearDecayEpsilonGreedy(
            start_epsilon, end_epsilon, decay_steps,
            env.random_action)

        opt = optimizers.Adam()
        opt.setup(q_func)



        # 优先经验回放
        rbuf = prioritized.PrioritizedReplayBuffer()

        phi = lambda x: x.astype(np.float32, copy=False)


        # 自己的DDQN
        agent = MyDoubleDQN(q_func,
                            opt,
                            rbuf,
                            gamma=0.99,
                            explorer=explorer,
                            replay_start_size=replay_start_size,
                            target_update_interval=10,  # target q网络多久和q网络同步
                            update_interval=update_interval,
                            phi=phi,
                            minibatch_size=minibatch_size,
                            gpu=args.gpu,  # 设置是否使用gpu
                            episodic_update_len=16)

        return agent


    def train():
        env = Env.MyEnv(state_size, action_size,
                        feature_max_count,
                        Detector('RandomForest'),
                        utils.DataLoader())
        agent = create_agent(env)
        train_agent(env, agent)

        # evaluate(env, agent)

        return env, agent

    train()

    # 用于计算本次训练中最大的准确率,及所选的特征以及训练过程中的平均准确率
    max_train_accuracy = max(train_accuracy)
    max_evaluate_accuracy = max(evaluate_accuracy)
    max_train_reward = max(train_reward)
    max_evaluate_reward = max(evaluate_reward)
    # 取索引
    max_train_accuracy_index = train_accuracy.index(max_train_accuracy)
    max_evaluate_accuracy_index = evaluate_accuracy.index(max_evaluate_accuracy)
    max_train_reward_index = train_reward.index(max_train_reward)
    max_evaluate_reward_index = evaluate_reward.index(max_evaluate_reward)
    # 找特征
    best_train_accuracy_feature = feature_list_train[max_train_accuracy_index]
    best_evaluate_accuracy_feature = feature_list_evaluate[max_evaluate_accuracy_index]
    best_train_reward_feature = feature_list_train[max_train_reward_index]
    best_evaluate_reward_feature = feature_list_train[max_evaluate_reward_index]

    # 统计训练过程中reward的平均值以及变化趋势
    average_reward = 0
    for i in range(len(train_reward) - 1):
        average_reward = average_reward + train_reward[i]
    average_reward = average_reward / len(train_reward)

    # 写入文件训练过程统计结果
    with open(result_file, 'a+') as f:
        f.write("Train reward:{}\n".format(train_reward))
        # 打印accuracy,用于绘图
        f.write("Train precision:{}\n".format(train_precision))
        # 打印recall,用于绘图
        f.write("Train recall:{}\n".format(train_recall))
        f.write("The max accuracy of the train:{}, the feature selected are:{}.\n".format(max_train_accuracy,
                                                                                          best_train_accuracy_feature))
        f.write("The max accuracy of the evaluate:{}, the feature selected are:{}.\n".format(max_evaluate_accuracy,
                                                                                             best_evaluate_accuracy_feature))
        f.write("The max reward of the train:{}, the feature selected are:{}\n".format(max_train_reward,
                                                                                       best_train_reward_feature))
        f.write("The max reward of the evaluate:{}, the feature selected are:{}\n".format(max_evaluate_reward,
                                                                                       best_evaluate_reward_feature))
        f.write("The average reward of this train:{}.\n".format(average_reward))



if __name__ == '__main__':

    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print("Time: {}".format(elapsed))
    # 训练时间
    with open(result_file, 'a+') as f:

        f.write("Training time:{} seconds".format(elapsed))
