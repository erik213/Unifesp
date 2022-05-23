import random
from datetime import datetime
random.seed(datetime.now())

class particleSwarmOptimization():
    def __init__(self, w, c1, c2, r1, r2, n):
        # Constants for update particles velocities
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.r1 = r1
        self.r2 = r2
        # Number of particles
        self.n = n
        # Arrays to store particles positions, velocities and best positions
        self.particlesPositions = []
        self.particlesVelocities = []
        self.particlesBestPositions = []

    def initializePositions(self, multiplier, subtrahend):
        for i in range(self.n):
            self.particlesPositions.append(
                multiplier * random.random() - subtrahend)

    def initializeVelocities(self, multiplier, subtrahend):
        for i in range(self.n):
            self.particlesVelocities.append(
                multiplier * random.random() - subtrahend)

    def initializeParticlesBestPositions(self):
        for i in range(self.n):
            self.particlesBestPositions.append(self.particlesPositions[i])

    def initializeGlobalBestPosition(self):
        self.globalBestPosition = self.particlesPositions[0]
        for i in range(1, self.n):
            if self.function(self.particlesPositions[i]) > self.function(self.globalBestPosition):
                self.globalBestPosition = self.particlesPositions[i]

    def initializeGlobalBestFitness(self):
        self.globalBestFitness = self.function(self.globalBestPosition)

    def initializeFunctionResults(self):
        self.functionResults = []
        for i in range(self.n):
            self.functionResults.append(self.function(self.particlesPositions[i]))

    # Function to be optimized
    # f(x) = 1 + (2*x) - (x)^2 
    def function(self, x):
        return 1+(2*x)-(x**2)

    # vi+1 = w*vi + c1*r1*(pBesti-xi) + c2*r2*(gBest-xi)
    def updateParticlesVelocities(self):
        for i in range(self.n):
            self.particlesVelocities[i] = self.w * self.particlesVelocities[i] + self.c1 * self.r1 * (
                self.particlesBestPositions[i] - self.particlesPositions[i]) + self.c2 * self.r2 * (self.globalBestPosition - self.particlesPositions[i])

    # xi+1 = xi + vi+1
    def updateParticlesPositions(self):
        for i in range(self.n):
            self.particlesPositions[i] += self.particlesVelocities[i]

    # if f(xi) > f(pBest)
    def updateParticlesBestPosition(self):
        for i in range(self.n):
            if self.function(self.particlesPositions[i]) > self.function(self.particlesBestPositions[i]):
                self.particlesBestPositions[i] = self.particlesPositions[i]

    # if f(xi) > f(gBest)
    def updateGlobalBestPosition(self):
        for i in range(self.n):
            if self.function(self.particlesPositions[i]) > self.function(self.globalBestPosition):
                self.globalBestPosition = self.particlesPositions[i]

    def updateGlobalBestFitness(self):
        self.globalBestFitness = self.function(self.globalBestPosition)

    def updateFunctionResults(self):
        for i in range(self.n):
            self.functionResults[i] = self.function(self.particlesPositions[i])


def main():
    n = 5
    iterations = 3
    pso = particleSwarmOptimization(0.70, 0.20, 0.60, 0.4657, 0.5319, n)
    pso.initializePositions(10, 0.5)
    pso.initializeVelocities(1, 0.5)
    pso.initializeGlobalBestPosition()
    pso.initializeGlobalBestFitness()
    pso.initializeParticlesBestPositions()
    pso.initializeFunctionResults()
    print("Iteration: 1")
    print("Positions = " + str(pso.particlesPositions))
    print("Velocities = " + str(pso.particlesVelocities))
    print("Results = " + str(pso.functionResults))
    print("Local Best Positions" + str(pso.particlesBestPositions))
    print("Global Best Fitness = " + str(pso.globalBestFitness))
    print("Global Best Position = " + str(pso.globalBestPosition) + "\n")
    for i in range(1, iterations):
        print("Iteration: " + str(i+1))
        pso.updateParticlesVelocities()
        pso.updateParticlesPositions()
        pso.updateParticlesBestPosition()
        pso.updateGlobalBestPosition()
        pso.updateGlobalBestFitness()
        pso.updateFunctionResults()
        print("Positions = " + str(pso.particlesPositions))
        print("Velocities = " + str(pso.particlesVelocities))
        print("Results = " + str(pso.functionResults))
        print("Local Best Positions" + str(pso.particlesBestPositions))
        print("Global Best Fitness = " + str(pso.globalBestFitness))
        print("Global Best Position = " + str(pso.globalBestPosition) + "\n")


if __name__ == "__main__":
    main()
