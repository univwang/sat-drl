import queue


class Sat:
    def __init__(self, A, V, f, kn):
        self.A = A
        self.kn = kn
        self.V = V
        self.f = f
        self.q_size = 0

    def q_update(self, A_all, dt, w):

        self.q_size = self.q_size + A_all - min(self.q_size + A_all, self.V * self.f * dt / w / self.kn)

        