import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


class Karm:
    def __init__(self, k):
        self.K = k
        # 设置随机种子
        # np.random.seed(0)
        # 从标准正态分布（mu=0，std=1）中取k个值作为每个臂的奖励均值
        self.k_arm_mu = np.random.standard_normal(k)

    def step(self, action):
        mu = self.k_arm_mu[action]
        reward = np.random.normal(mu, 1, 1)  # 从正态分布中抽取1个随机样本
        return reward

    def plot_karm(self):
        # 画出生成的k个臂各自的分布曲线
        print(f"k个臂的奖励正态分布均值为： {self.k_arm_mu}")
        # 显示每个动作的奖励所在的分布
        for k_arm_num in range(self.K):
            plt.subplot(2, 5, k_arm_num + 1)
            mu = self.k_arm_mu[k_arm_num]
            sigma = 1
            s = np.random.normal(mu, sigma, 1000)
            count, bins, ignored = plt.hist(s, 20, density=True, stacked=True)
            plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
                     linewidth=3, color='y')  # 画出标准的正态分布曲线
        plt.show()


class Agent:
    def __init__(self, policy, Q_update_method='Sample_average', epsilon=0.01, fix_step=0.1, ucb_c=2, temperature=0.1):
        """
        :param policy: 'egreedy'为e-greedy贪婪选择, 'ucb'为置信度上界upper confidence bound, 'gradient'梯度赌博机算法
        :param Q_update_method: 'Sample_average'采样平均法, 'fix_step'固定步长alpha
        :param epsilon: e-greedy贪婪选择的epsilon
        :param fix_step:
        :param ucb_c:
        :param temperature: 怪,这啥
        """
        self.settings = Settings()
        self.k = self.settings.k
        # 具体到某一种时其实只用留下来一个
        self.Q_estimate = np.zeros(self.k)  # Q动作价值函数
        self.Q_ucb = np.zeros(self.k)  # ucb估计的动作价值函数 upper confidence bound
        self.H_gradient = np.zeros(self.k)  # 梯度赌博机算法的偏好函数

        self.N = np.zeros(self.k)  # 每个动作被选择的次数

        self.policy = policy  # 选择动作采取的策略
        self.Q_update_method = Q_update_method  # 更新动作价值的方法

        self.fix_step = fix_step  # 固定步长大小 alpha
        self.ucb_c = ucb_c  # ucb参数c
        self.epsilon = epsilon  # 贪心探索概率
        self.temperature = temperature

        self.count = 0  # 每一轮交互的次数，新的一轮次数会清零
        self.Q_total = 0  # 当前获得了多少奖励，新的一轮次数会清零
        self.reward_record = [0]  # 记录每一步的奖励值
        self.action_record = [0]  # 记录每一步的动作
        self.Q_total_record = [0]  # 记录每一步动作下的总奖励值,怪,没用上

    def cal_gradient_pi(self):
        exp = np.exp(self.H_gradient)
        summary = np.sum(exp)
        pi = exp/summary
        return pi

    def reset(self):
        self.Q_estimate = np.zeros(self.k)  # Q动作价值函数
        self.Q_ucb = np.zeros(self.k)  # ucb估计的动作价值函数
        self.H_gradient = np.zeros(self.k)  # 梯度赌博机算法的偏好函数
        self.N = np.zeros(self.k)  # 每个动作被选择的次数

        self.count = 0  # 每一轮交互的次数，新的一轮次数会清零
        self.Q_total = 0  # 当前获得了多少奖励，新的一轮次数会清零
        self.reward_record = [0]  # 记录每一步的奖励值
        self.action_record = [0]  # 记录每一步的动作
        self.Q_total_record = [0]  # 记录每一步动作下的总奖励值,怪,没用上

    def choose_action(self):
        if self.policy == 'egreedy':  # e-greedy贪婪选择
            if np.random.rand() < self.epsilon:  # 随机取值 [0, 1)
                return np.random.choice(self.k)
            else:
                return np.random.choice(np.where(self.Q_estimate == np.max(self.Q_estimate))[0])

        elif self.policy == 'ucb':
            if 0 in self.N:  # 先需要遍历所有操作,怪
                action = np.where(self.N == 0)[0][0]
            else:
                action = np.random.choice(np.where(self.Q_ucb == np.max(self.Q_ucb))[0])
            return action

        elif self.policy == 'gradient':
            pi = self.cal_gradient_pi()
            action = np.random.choice(np.arange(len(pi)), p=pi)
            return action

    def update(self, action, reward):
        self.count += 1  # 交互次数加1
        self.N[action] += 1  # 被选择动作次数加1
        self.Q_total += reward
        if self.policy == 'egreedy' or self.policy == 'ucb':
            if self.Q_update_method == 'Sample_average':  # 采样平均法 alpha=1/n
                self.Q_estimate[action] += (reward - self.Q_estimate[action]) / self.N[action]
            elif self.Q_update_method == 'fix_step':  # 固定步长法 alpha固定
                self.Q_estimate[action] += self.fix_step * (reward - self.Q_estimate[action])
            self.Q_ucb[action] = self.Q_estimate[action] + self.ucb_c * np.sqrt(np.log(self.count) / (self.N[action]))
        elif self.policy == 'gradient':
            pi = self.cal_gradient_pi()
            self.H_gradient -= self.fix_step * pi * (reward - self.Q_total / self.count)
            self.H_gradient[action] += self.fix_step * (reward - self.Q_total/self.count)
            # for i in range(self.k):
            #     if i == action:
            #         self.H_gradient[i] += self.fix_step * (1 - pi[i]) * (reward - self.Q_total/self.count)
            #     else:
            #         self.H_gradient[i] -= self.fix_step * pi[i] * (reward - self.Q_total / self.count)


def play(agent_list):
    rewards = np.zeros((len(agent_list), set_para.play_epochs, set_para.play_steps))  # 构建三维数据存储所有的动作奖励
    for agent_num, agent in enumerate(agent_list):  # 选择不同的智能体（不同策略）
        for play_epoch in trange(set_para.play_epochs):  # 对独立的多台赌博机进行实验
            karms = Karm(set_para.k)  # 每一轮都实例化一个新的k臂赌博机
            # print(f"k个臂的奖励正态分布均值为： {karms.k_arm_mu}")
            agent.reset()  # 重置智能体状态
            for play_step in range(set_para.play_steps):  # 对单独的一台赌博机进行多次交互实验
                action = agent.choose_action()  # 智能体选择一个动作，与环境进行交互
                reward = karms.step(action)  # 环境返回刚才动作的奖励
                agent.update(action, reward)  # 智能体根据动作和相应的动作价值更新动作价值表
                rewards[agent_num, play_epoch, play_step] = reward  # 储存奖励
    return rewards


class Settings:
    def __init__(self):
        self.k = 10  # 设置赌博机的臂数量
        self.play_epochs = 2000  # 设置独立的赌博机台数
        self.play_steps = 1000  # 设置每次交互学习的次数
        self.print_every_step = 1000  # 设置每交互几次，打印平均的学习结果


if __name__ == '__main__':
    set_para = Settings()  # 相关参数装订
    # 定义智能体列表，将不同策略的智能体装载在一个列表中，分别进行实验
    agent_list = []
    agent_list.append(Agent(policy='egreedy', Q_update_method='Sample_average', epsilon=0.0))
    agent_list.append(Agent(policy='egreedy', Q_update_method='Sample_average', epsilon=0.01))
    agent_list.append(Agent(policy='egreedy', Q_update_method='Sample_average', epsilon=0.1))
    agent_list.append(Agent(policy='ucb', Q_update_method='Sample_average', ucb_c=2))
    agent_list.append(Agent(policy='gradient'))
    rewards = play(agent_list)
    rewards_mean_plot = rewards.mean(axis=1)  # 将2000次的仿真结果求平均

    plt.figure(1)  # 画图这里不是自动生成的，具体参数需根据具体仿真工况进行设置
    plt.plot(rewards_mean_plot[0, :], label='$ greedy,\epsilon = %.02f $' % 0.0)
    plt.plot(rewards_mean_plot[1, :], label='$ greedy,\epsilon = %.02f $' % 0.01)
    plt.plot(rewards_mean_plot[2, :], label='$ greedy,\epsilon = %.02f $' % 0.1)
    plt.plot(rewards_mean_plot[3, :], label='$ ucb,c = %.02f $' % 2)
    plt.plot(rewards_mean_plot[4, :], label='$ gradient,\\alpha = %.02f $' % 0.1)
    plt.xlabel("step")
    plt.ylabel("average reward")
    plt.legend()
    plt.savefig('./figure_2000_1000.png')

    # agent_list = []
    # agent_list.append(Agent(policy='gradient'))
    # rewards = play(agent_list)
    # rewards_mean_plot = rewards.mean(axis=1)  # 将2000次的仿真结果求平均
    #
    # plt.figure(1)  # 画图这里不是自动生成的，具体参数需根据具体仿真工况进行设置
    # plt.plot(rewards_mean_plot[0, :], label='$ gradient,\\alpha = %.02f $' % 0.1)
    # plt.xlabel("step")
    # plt.ylabel("average reward")
    # plt.legend()
    # plt.savefig('./1.png')
