#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''

@File    :   agent.py    
@Author  :   你牙上有辣子
@Contact :   johnsogunn23@gmail.com
@Create Time : 2021-10-15 14:39     
@Discription :

实现智能体的一些操作

'''

# import lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from chainer import cuda
from future import standard_library

standard_library.install_aliases()

import chainer

from chainerrl.agents import double_dqn


# 继承chainer的DDQN
class MyDoubleDQN(double_dqn.DoubleDQN):

    def act(self, state, action_list):
        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                action_value = self.model(
                    self.batch_states([state], self.xp, self.phi))

                # 设置当前状态的state，保证在action_value选取动作的时候考虑一下目前已经选了的state
                # 此处不能直接写action_value.load_current_state(state)
                # 应该使用self.batch_states，保证在CPU和GPU中都能使用
                action_value.load_current_action(
                    action_list
                )
                q = float(action_value.max.data)
                action = cuda.to_cpu(action_value.greedy_actions_with_state.data)[0]

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        # paper2的返回值
        # return action, action_value.q_values.data.astype(np.float)
        # chanierrl的返回
        return action

    def act_and_train(self, state, reward, action_list):

        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                action_value = self.model(
                    self.batch_states([state], self.xp, self.phi))

                # 设置当前状态的state，保证在action_value选取动作的时候考虑一下目前已经选了的state
                # 此处不能直接写action_value.load_current_state(state)
                # 应该使用self.batch_states，保证在CPU和GPU中都能使用
                action_value.load_current_action(
                    action_list
                )
                q = float(action_value.max.data)
                greedy_action = cuda.to_cpu(action_value.greedy_actions_with_state.data)[
                    0]

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q


        action = self.explorer.select_action(
            self.t, lambda: greedy_action, action_value=action_value)
        self.t += 1


        # 更新目标网络
        if self.t % self.target_update_interval == 0:
            self.sync_target_network()

        if self.last_state is not None:
            assert self.last_action is not None
            # 向经验回放池中放入数据
            self.replay_buffer.append(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=state,
                next_action=action,
                is_state_terminal=False)

        self.last_state = state
        self.last_action = action

        self.replay_updater.update_if_necessary(self.t)

        # paper2的返回
        # return self.last_action, action_value.q_values.data.astype(np.float), greedy_action
        # chainerrl的返回
        return self.last_action