import numpy as np

from Network import Network
from Sat import Sat


class env:
    def __init__(self):
        self.dt = 120  # 一个时隙的时间
        self.T = 100  # 一个回合的的长度
        self.N = 20  # 网络的节点数量
        self.p = 10  # 传输功率
        self.kn = 10  # 任务的比特数
        self.px = 0.5  # 未来数据占的比例
        self.t = 0  # 当前的时间点
        self.sats = [Sat(10, 10, 10) for i in range(self.N)]
        self.net = Network(self.N)
        self.A = [0 for i in range(self.N)]
        # 一个任务来临的时间序列
        self.line = np.linspace(1, 100, self.T)
        self.w = 10

    def ini(self):
        self.A = [0 for i in range(self.N)]
        self.sats = [Sat(10, 10, 10) for i in range(self.N)]


    def reset(self):
        self.ini()
        # self.update()

        return

    def update(self, action=None):

        if action is None:
            action = []

        for i in range(self.N):
            self.sats[i].q_update(self.A[i], self.dt, self.w)

        R = []
        for key in self.net.L:
            R.append(self.net.L[key])
        Q = [self.sats[i].q_size for i in range(len(self.sats))]
        A = self.A
        H = self.line
        next_state = R + Q + A + H

        return next_state

    def check(self, action):
        for i in range(self.N):
            if self.sats[i].q_size < 0:
                return False

        for i in range(self.N):
            if self.A[i] != self.line[self.t]:
                return False

    def step(self, action):

        q = 0
        for i in range(self.N):
            for j in range(self.N):
                q += action[j][i]
            self.A[i] = q

        reward = self.reward(action)
        next_state = self.update(action)

        done = self.check(action) or self.t == self.T
        if done is True:
            reward -= 1000
        self.t += 1
        return done, next_state, reward

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
            for j in range(len(self.line)):
                pred = pred * self.px + self.line[j]
            b_all.append((self.sats[i].V * self.sats[i].f / self.w)
                         * (self.sats[i].q_size + self.A[i] + pred))

        b_ = sum(b_all) / self.N

        res = 0
        for b in b_all:
            res += (b - b_) ** 2

        return res / self.N

    def reward(self, action):

        return self.get_dt(action) + self.get_dc(action) + self.get_db() + self.get_dw()

