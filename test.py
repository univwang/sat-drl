from DataG import Generator
from env import env
import numpy as np


myenv = env()
state = myenv.reset()
# 动作为从该处调度到其他节点的任务数量
action = [[1, 0], [1, 0], [0, 0]]

done, reward, next_state = myenv.step(action)

print(done, reward, next_state)