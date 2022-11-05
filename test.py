from DataG import Generator
from env import env
import numpy as np


# myenv = env()
# state = myenv.reset()
# # 动作为从该处调度到其他节点的任务数量
# action = [[1, 0], [1, 0], [0, 0]]
#
# done, reward, next_state = myenv.step(action)

line = [1, 3, 10, 11, 11, 8, 3, 6, 9, 4]
line = [line for i in range(3)]
g = Generator(line[0])
g.train()
G = g.get_predict()