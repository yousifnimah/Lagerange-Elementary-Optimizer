import numpy as np


class Population:
    def __init__(self, size):
        self.data = []
        self.size = size

    def initialize(self, lowB, upB, numGs):
        for _ in range(self.size):
            pos = np.random.uniform(lowB, upB, numGs)
            self.data.append(pos)


    def loadDataFromFile(self, filename):
        # here you will be able to read datasets from file to be stored in self.pop
        pass

