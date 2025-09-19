from datasets import Population
from lagrange_elementary_optimizer import LEO

# Data Calibration
lowB = -5
upB = 15
COPer = 0.6


# General Calibration
epochs = 50
popSize = 100
num_g=2
task_times = None

# Initialize datasets
population = Population(size=popSize)
population.initialize(lowB, upB, num_g)

# TODO: population.loadDataFromFile() -> this function will be used to read data from a file

# Initialize LEO
leo = LEO(epochs, popSize, num_g, task_times,lowB, upB, COPer, population.data)
leo.learn()

# Print Best Solution
print(f"Best Solution: {leo.best_sol}")

# Display Plot Figure Graph of best costs
leo.plotBestCosts()