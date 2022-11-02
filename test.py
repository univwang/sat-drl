from DataG import Generator
import numpy as np

mydata = np.linspace(1, 30, 30) + 10 * np.random.random(30) + 0

g = Generator(mydata)

g.train()
g.get_forest(30)
g.draw()
