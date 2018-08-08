import sys
import numpy

def parseTrainFile(fileName):
    fileStream = open(fileName, "r")
    tokens = [line.split() for line in fileStream]
    dimensionality = tokens[0][1]
    pointCount = tokens[0][0]
    tokens.remove(tokens[0])
    points = []
    labels = []
    for line in tokens:
        points.append([float(line[x]) for x in range(len(line) - 1)])
        labels.append(int(line[len(line)-1]))
    return points, labels

def parseTestFile(fileName):
    fileStream = open(fileName, "r")
    tokens = [line.split() for line in fileStream]
    dimensionality = tokens[0][1]
    pointCount = tokens[0][0]
    tokens.remove(tokens[0])
    points = []
    labels = []
    for line in tokens:
        points.append([float(line[x]) for x in range(len(line))])
    return points

def kNN(x, trainingPoints, trainingLabel, k):
    #brute force solution
    distances = [(numpy.linalg.norm(numpy.matrix(x) - numpy.matrix(trainingPoints[i])), trainingLabel[i]) for i in range(len(trainingPoints))]
    distances.sort()
    voters = [distances[i][1] for i in range(k)]
    votes = numpy.bincount(voters)
    majorityVote = 0
    for vote in votes:
        if vote > majorityVote:
            majorityVote = vote
    for voter in voters:
        if(votes[voter] == majorityVote):
            return voter



k = int(sys.argv[1])
trainingPoints, trainingLabels = parseTrainFile(sys.argv[2])
testingPoints = parseTestFile(sys.argv[3])
testResults = [kNN(point, trainingPoints, trainingLabels, k) for point in testingPoints]

for i in range(len(testResults)):
    print(str(i+1) + ". ", end="")
    [print(" " + str(x), end="") for x in testingPoints[i]]
    print(" -- " + str(testResults[i]))

