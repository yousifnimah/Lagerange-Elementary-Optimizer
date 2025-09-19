import numpy as np
from matplotlib import pyplot as plt


class LEO:
    def __init__(self, epochs, pop_size, num_g, task_times,lowB, upB, COPer, data):
        self.numIterate = epochs
        self.popSize = pop_size
        self.numGs = num_g

        self.task_times = task_times
        self.lowB = lowB
        self.upB = upB
        self.COPer = COPer

        # LEO Calibration
        self.MutPer = 0.3
        self.muRate = 0.02
        self.dp = 0.5
        self.beta = 10


        # initialize the population by calculating cost per each position
        self.pop = [{"cost": self.cost_function(pos), "position": pos} for pos in data]

        self.best_sol = self.pop[0]
        self.best_costs = []


    # ================================
    # Main LEO Function
    # ================================
    def learn(self):
        # Sort population
        self.pop.sort(key=lambda x: x['cost'])


        numOfMutate = int(np.round(self.MutPer * self.popSize))
        numParent = 2 * int(np.round(self.COPer * self.popSize / 2))

        for it in range(self.numIterate):  # epochs
            costs = np.array([ind['cost'] for ind in self.pop])
            worstCost = max(costs)
            P = np.exp(-self.beta * costs / worstCost)
            P = P / np.sum(P)

            # Identify populations
            fPop, wPop, gPop = self.identify_population()
            val = self.find_value(len(fPop))
            val = int(val / 2)
            fPopindex = []

            # Offspring generation
            popc2 = []
            numParentCheck = 1
            finCount = 0

            for c in range(val // 2):
                i1 = self.parent_selection(fPop, fPopindex)
                i2 = self.parent_selection(fPop, fPopindex)
                fPopindex.extend([i1, i2])
                parent1 = fPop[i1]
                parent2 = fPop[i2]
                # Crossover
                child1_pos, child2_pos = self.crossover(parent1['position'], parent2['position'], self.lowB, self.upB)
                popc2.append({'position': child1_pos, 'cost': self.cost_function(child1_pos)})
                popc2.append({'position': child2_pos, 'cost': self.cost_function(child2_pos)})
                numParentCheck += 1

            # Mutation
            for _ in range(numOfMutate):
                i = np.random.randint(len(popc2))
                p = popc2[i]
                mutated_pos = self.mutate(p['position'], self.muRate, self.lowB, self.upB)
                popc2[i] = {'position': mutated_pos, 'cost': self.cost_function(mutated_pos)}

            # Merge populations
            self.pop.extend(popc2)
            self.pop.sort(key=lambda x: x['cost'])
            self.pop = self.pop[:self.popSize]

            self.best_sol = self.pop[0]
            self.best_costs.append(self.best_sol['cost'])
            print(f"Epoch {it + 1}, Best Cost = {self.best_sol['cost']:.3f}")
        print("Training is completed\n")


    def plotBestCosts(self):
        # Plot
        plt.figure()
        plt.plot(self.best_costs, linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Best Cost")
        plt.title("LEO Optimization Convergence")
        plt.show()

    # ================================
    # Cost Function (example for tasks scheduling)
    # ================================
    def cost_function(self, position):
        """
        Compute cost of a solution.
        If task_times is None, generate random cost as example.
        position: array-like, represents assignment of tasks to resources
        """
        if self.task_times is None:
            return np.sum(position ** 2)  # simple benchmark
        else:
            # For real scheduling, compute makespan
            assigned = self.task_times[np.arange(len(position)), position.astype(int)]
            return np.sum(assigned)

    # ================================
    # Crossover Function
    # ================================
    def crossover(self, x1, x2, lowB, upB):
        alpha = 0.2 + (0.3 - 0.2) * np.random.rand()
        l1 = (x1 - x2) ** 2 + (x2 - 1) ** 2 - alpha * (x1 + 2 * x2 - 1) - alpha * (2 * x1 + x2 - 1)
        l2 = (x2 - x1) ** 2 + (x1 - 1) ** 2 - alpha * (x2 + 2 * x1 - 1) - alpha * (2 * x2 + x1 - 1)
        y1 = l2 / x1
        y2 = l1 / x2
        y1 = np.clip(y1, lowB, upB)
        y2 = np.clip(y2, lowB, upB)
        return y1, y2

    # ================================
    # Mutation Function
    # ================================
    def mutate(self, x, muRate, lowB, upB):
        numGs = len(x)
        mutateNumber = max(1, int(np.ceil(muRate * numGs)))
        j = np.random.choice(numGs, mutateNumber, replace=False)
        sigma = 0.01 * np.random.uniform(-1, 1, size=j.shape) * (upB - lowB)
        y = x.copy()
        y[j] += sigma * np.random.randn(len(j))
        y = np.clip(y, lowB, upB)
        return y

    # ================================
    # Parent Selection Function
    # ================================
    def parent_selection(self, fPop, fPopindex):
        i1 = np.random.randint(len(fPop))
        while i1 in fPopindex:
            i1 = np.random.randint(len(fPop))
        return i1

    # ================================
    # Ensure even number function
    # ================================
    def find_value(self, val):
        if val % 4 == 0:
            return val
        else:
            return val + 2

    # ================================
    # IdentifyPopulation Function
    # ================================
    def identify_population(self):
        popSize = len(self.pop)
        sr = int(np.floor(popSize * self.dp))
        sorted_indices = np.argsort([ind['cost'] for ind in self.pop])
        halfsort = sorted_indices[:sr]
        fPop = [self.pop[i] for i in halfsort]  # first population

        worstOf1stGroup = self.pop[halfsort[-1]]
        bestIndiv = self.pop[halfsort[0]]

        # Better population
        betterPop = [ind for ind in self.pop if ind['cost'] < worstOf1stGroup['cost']]
        # Worst population
        worstPop = [ind for ind in self.pop if ind['cost'] >= worstOf1stGroup['cost']]
        # Good population
        gPop = [ind for ind in betterPop if ind['cost'] >= bestIndiv['cost']]

        return fPop, worstPop, gPop
