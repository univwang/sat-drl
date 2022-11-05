import numpy as np
from DataG import Generator

from Network import Network
from Sat import Sat


class env:
    def __init__(self):
        self.dt = 120  # 一个时隙的时间
        self.T = 100  # 一个回合的的长度
        self.N = 3  # 网络的节点数量
        self.p = 10  # 传输功率
        self.kn = 10  # 任务的比特数
        self.px = 0.5  # 未来数据占的比例
        self.t = 0  # 当前的时间点
        self.sats = [Sat(10, 10, 10) for i in range(self.N)]
        self.net = Network(self.N)
        self.A = [0 for i in range(self.N)]  # 任务分配方案
        # 一个任务来临的时间序列
        # self.line = np.linspace(1, 30, self.T) + 10 * np.random.random(self.T)
        # 任务多一点
        self.line = [1, 3, 10, 11, 11, 8, 3, 6, 9, 4]
        # self.line = np.expand_dims(self.line, 0).repeat(self.N, axis=0)
        self.line = [self.line for i in range(self.N)]
        self.g = Generator(self.line[0])
        self.g.train()
        self.G = self.g.get_predict()
        self.w = 10

    def ini(self):
        self.A = [0 for i in range(self.N)]
        self.sats = [Sat(10, 10, 10) for i in range(self.N)]

    def reset(self):
        self.ini()
        R = []
        for key in self.net.L:
            R.append(self.net.L[key])
        Q = [self.sats[i].q_size for i in range(len(self.sats))]
        A1 = [self.line[i][self.t] for i in range(self.N)]
        H = []
        for i in range(self.N):
           H.append(self.line[i][self.t + 1: self.t + 3])

        H = sum(H, [])  # 二维展开
        H = list(map(int, H))  # 转化为整数
        return np.concatenate((np.array(R), np.array(Q), np.array(A1), np.array(H)))

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
           H.append(self.line[i][self.t + 1: self.t + 3])

        H = sum(H, [])  # 二维展开
        H = list(map(int, H))  # 转化为整数
        next_state = np.concatenate((R, Q, A1, H))
        return next_state

    def check(self, env_action):
        for i in range(self.N):
            if self.sats[i].q_size > self.sats[i].A or env_action[i][i] < 0:
                return True
        return False

    def step(self, action):

        env_action = [[0 for i in range(self.N)] for i in range(self.N)]
        # action 是除了自己以外的调度任务数量

        for i in range(self.N):
            for j in range(self.N - 1):
                if j >= i:
                    env_action[i][j + 1] = action[i][j]
                else:
                    env_action[i][j] = action[i][j]


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
        done = self.check(env_action) or self.t == self.T
        if done is True:
            reward -= 10000
        return done, reward, next_state

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
                d += action[i][j] * self.kn * self.w / self.sats[j].f
        return d

    def get_db(self):
        d = 0
        for i in range(self.N):
            d += self.w * self.sats[i].q_size / self.sats[i].V / self.sats[i].f
        return d

    def get_dw(self, action):
        d = 0
        for i in range(self.N):
            d += self.kn * self.w * (self.A[i] - action[i][i]) / (2 * self.sats[i].V * self.sats[i].f)
        return d

    def get_load_b(self):
        b_all = []
        for i in range(self.N):
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
        return -(self.get_dt(action) + self.get_dc(action) + self.get_db() + self.get_dw(action) + self.get_load_b())
