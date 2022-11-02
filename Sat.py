import queue


class Sat:
    def __init__(self, A, V, f):
        self.A = A
        self.V = V
        self.f = f
        self.q_size = A

    def q_update(self, A_all, dt, w):

        self.q_size = self.q_size + A_all - max(self.q_size + A_all, self.V * self.f * dt / w)

        