import sys
import numpy

def parseFile(fileName):
    fileStream = open(fileName, "r")
    tokens = [line.split() for line in fileStream]
    dimensionality = tokens[0][1]
    pointCount = tokens[0][0]
    tokens.remove(tokens[0])
    return [[float(y) for y in x] for x in tokens]

def mahalanobisDistance(x, centroid, sigmaInv):
    return numpy.sqrt(numpy.matmul(numpy.transpose(x-centroid), numpy.matmul(sigmaInv, (x-centroid))))

trainingFileName = sys.argv[1]
testingFileName = sys.argv[2]

trainingPoints = parseFile(trainingFileName)
testingPoints = parseFile(testingFileName)

trainingAverage = numpy.mean(trainingPoints, axis=0)
xPrime = numpy.subtract(trainingPoints, trainingAverage)
S = numpy.matmul(numpy.transpose(xPrime), xPrime)
S = S/len(trainingPoints)
sigmaInverse = numpy.linalg.inv(S)

print("Centroid:")
[print(x, end=" ") for x in trainingAverage]
print("")
print("Covariance Matrix:")
for row in S:
    [print(element, end=" ") for element in row]
    print("")
print("Distances:")
for k in range(len(testingPoints)):
    print(str(k + 1) + ". ", end="")
    [print(str(testingPoints[k][dim]) + "   ", end="") for dim in range(len(testingPoints[k]))]
    print(" -- " + str(
        round(mahalanobisDistance(testingPoints[k], trainingAverage, sigmaInverse),4)))
