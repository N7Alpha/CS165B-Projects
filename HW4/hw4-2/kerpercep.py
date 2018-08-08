import sys
import numpy

def parseFile(fileName):
    fileStream = open(fileName, "r")
    tokens = [line.split() for line in fileStream]
    dimensionality = tokens[0][1]
    pointCount = tokens[0][0]
    tokens.remove(tokens[0])
    return [[float(y) for y in x] for x in tokens]

def gaussianKernel(x1, x2, bandwidth):
    return numpy.exp(-numpy.power(numpy.linalg.norm(numpy.matrix(x1) - numpy.matrix(x2))/(2*bandwidth*bandwidth),2))

bandwidth = 1
class Perceptron:

    k = gaussianKernel
    alpha = None
    data = None
    labels = None


    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.train(self.k, data, labels)

    def classify(self, point):
        if numpy.sum([self.alpha[j]*self.labels[j]*gaussianKernel(point, self.data[j], bandwidth) for j in range(len(self.data))]) > 0:
            return 1
        else:
            return -1

    def train(self, kernel, data, labels):
        self.alpha = [0 for x in range(len(data))]
        converged = False
        while converged == False:
            converged = True
            for i in range(len(data)):
                if labels[i] * numpy.sum([self.alpha[j]*labels[j]*gaussianKernel(data[i], data[j], bandwidth) for j in range(len(data))]) <= 0:
                    self.alpha[i] = self.alpha[i] + 1
                    converged = False

bandwidth = float(sys.argv[1])
positiveTrainingPoints = parseFile(sys.argv[2])
negativeTrainingPoints = parseFile(sys.argv[3])
positiveTestingPoints = parseFile(sys.argv[4])
negativeTestingPoints = parseFile(sys.argv[5])


trainingPoints = positiveTrainingPoints + negativeTrainingPoints
trainingLabels = [1 for x in positiveTrainingPoints] + [-1 for x in negativeTrainingPoints]

classifier = Perceptron(data=trainingPoints, labels=trainingLabels)
positiveTestResults = [classifier.classify(point) for point in positiveTestingPoints]
negativeTestResults = [classifier.classify(point) for point in negativeTestingPoints]

falsePositives = negativeTestResults.count(1)
falseNegatives = positiveTestResults.count(-1)

print("Alphas:", end="")
[print("  " + str(a), end="") for a in classifier.alpha]
print("")
print("False positives: " + str(falsePositives))
print("False negatives: " + str(falseNegatives))
print("Error rate: " + str((falsePositives+falseNegatives)/(len(positiveTestingPoints)+len(negativeTestingPoints))))

