import numpy as np
from DataG import Generator

from Network import Network
from Sat import Sat


class env:
    def __init__(self):
        self.dt = 120  # 一个时隙的时间
        self.T = 5  # 一个回合的的长度
        self.N = 3  # 网络的节点数量
        self.p = 10  # 传输功率
        self.kn = 1  # 任务的比特数
        self.px = 0.41  # 未来数据占的比例
        self.t = 0  # 当前的时间点
        # self.sats = [Sat(10, 10, 10) for i in range(self.N)]
        self.sats = [Sat(0, 0, 0, self.kn), Sat(0, 0, 0, self.kn), Sat(2000, 32, 12.8, self.kn)]
        self.net = Network(self.N)

        self.A = [0 for i in range(self.N)]  # 任务分配方案
        # 一个任务来临的时间序列
        # self.line = np.linspace(1, 30, self.T) + 10 * np.random.random(self.T)
        self.line = [100, 300, 500, 500, 600, 800, 300, 600, 700, 400]
        # self.line = np.expand_dims(self.line, 0).repeat(self.N, axis=0)
        self.line = [self.line for i in range(self.N)]
        self.G = [Generator(self.line[i]) for i in range(self.N)]
        for g in self.G:
            g.train()
        self.w = 8

    def ini(self):
        self.A = [0 for i in range(self.N)]
        # self.sats = [Sat(10, 10, 10) for i in range(self.N)]
        self.line = np.random.randint(100, 1000, 10)
        self.line = list(self.line)
        self.line = [self.line for i in range(self.N)]
        self.sats = [Sat(0, 0, 0, self.kn), Sat(0, 0, 0, self.kn), Sat(2000, 32, 12.8, self.kn)]

    def reset(self):
        self.ini()

        self.t = 0
        R = []
        for key in self.net.L:
            R.append(self.net.L[key])
        Q = [self.sats[i].q_size for i in range(len(self.sats))]
        A1 = [self.line[i][self.t] for i in range(self.N)]
        H = []
        for i in range(self.N):
            H.append(self.line[i][self.t + 1: self.t + 3])  # 两组后边的数据

        H = sum(H, [])  # 二维展开
        H = list(map(int, H))  # 转化为整数

        return np.concatenate([np.array(R), np.array(Q), np.array(A1), np.array(H)], axis=0)

    def update(self):
        for i in range(self.N):
            self.sats[i].q_update(self.A[i], self.dt, self.w)

        R = []
        for key in self.net.L:
            R.append(self.net.L[key])
        Q = [self.sats[i].q_size for i in range(len(self.sats))]
        A1 = [self.line[i][self.t] for i in range(self.N)]

        H = []
        for i in range(self.N):
            H.append(self.G[i].get_predict(self.t + 1, self.t + 2))  # 两组后边的数据

        H = sum(H, [])  # 二维展开
        H = list(map(int, H))  # 转化为整数

        next_state = np.concatenate((R, Q, A1, H))
        return next_state

    def check(self, env_action):
        reward = 0

        for i in range(self.N):
            for j in range(self.N):
                if env_action[i][j] < 0:
                    reward += env_action[i][j]

        # if reward < 0:
        #     return reward
        if reward > 0:
            reward = 0

        for i in range(self.N):
            if self.sats[i].q_size > self.sats[i].A:
                reward -= self.sats[i].q_size - self.sats[i].A

        return reward

    def step(self, action):
        action = list(map(round, action))
        env_action = [[0 for i in range(self.N)] for i in range(self.N)]
        # action 是除了自己以外的调度任务数量
        publish = 0
        get = 0

        for i in range(self.N):
            for j in range(self.N - 1):
                if j >= i:
                    env_action[i][j + 1] = action[i * (self.N - 1) + j]
                else:
                    env_action[i][j] = action[i * (self.N - 1) + j]

        # print(env_action)
        # 遍历卫星，如果发现有调度的数量大于到来的任务数量，惩罚并归0

        for i in range(self.N):
            q = 0
            for j in range(self.N):
                if q + env_action[i][j] > self.line[i][self.t]:
                    # publish += self.line[i][self.t] - (q + env_action[i][j])
                    env_action[i][j] = 0
                q += env_action[i][j]
            # if q > self.line[i][self.t]:
            #     publish += self.line[i][self.t] - q
            get += q

        for i in range(self.N):
            q = 0
            for j in range(self.N):
                q += env_action[j][i]
            self.A[i] = q

        for i in range(self.N):
            q = 0
            for j in range(self.N):
                q += env_action[i][j]
            env_action[i][i] = self.line[i][self.t] - q
            self.A[i] += env_action[i][i]

        self.t += 1
        reward = self.reward(env_action)
        next_state = self.update()

        # # 遍历卫星，如果发现队列超了，惩罚
        # for i in range(self.N):
        #     if self.sats[i].q_size > self.sats[i].A:
        #         publish += self.sats[i].A - self.sats[i].q_size
        #         self.sats[i].q_size = self.sats[i].A

        if self.t == self.T:
            isdone = True
        else:
            isdone = False

        reward += publish * 5 + get
        return isdone, reward, next_state

    def get_et(self, action):

        e = 0
        for i in range(self.N):
            for j in range(self.N):
                e += action[i][j] * self.p * self.net.get_l(i, j)

        return e

    def get_ec(self, action):

        e = 0
        for i in range(self.N):
            for j in range(self.N):
                e += action[i][j] * (self.sats[j].f ** 2) * self.kn * self.w

        return

    def get_dt(self, action):

        d = 0
        for i in range(self.N):
            for j in range(self.N):
                d += action[i][j] * self.kn * self.net.get_l(i, j)

        return d

    def get_dc(self, action):
        d = 0
        for i in range(self.N):
            for j in range(self.N):
                if self.sats[j].f > 0:
                    d += action[i][j] * self.kn * self.w / self.sats[j].f
        return d

    def get_db(self):
        d = 0
        for i in range(self.N):
            if self.sats[i].f > 0:
                d += self.w * self.sats[i].q_size * self.kn / self.sats[i].V / self.sats[i].f
        return d

    def get_dw(self, action):
        d = 0
        for i in range(self.N):
            if self.sats[i].f > 0:
                d += self.kn * self.w * (self.A[i] - action[i][i]) / (2 * self.sats[i].V * self.sats[i].f)
        return d

    def get_load_b(self):
        b_all = []
        for i in range(self.N):
            if self.sats[i].f > 0:
                pred = 0
                for j in range(len(self.line[i])):
                    pred = pred * self.px + self.line[i][j]
                b_all.append((self.sats[i].V * self.sats[i].f / self.w)
                             * (self.sats[i].q_size + self.A[i] + pred))

        b_ = sum(b_all) / self.N

        res = 0
        for b in b_all:
            res += (b - b_) ** 2

        return res / self.N

    def reward(self, action):
        return 0
        # print(self.get_dt(action), self.get_dc(action), self.get_db(), self.get_dw(action), self.get_load_b())
        return -(self.get_dt(action) / 1000 + self.get_dc(action) / 10 + self.get_db() + self.get_dw(
            action) + self.get_load_b() / 1000)
