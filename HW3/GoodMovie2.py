import sys
from sklearn import tree
import numpy

def parseFile(fileName) :
    fileStream = open(fileName, "r")
    tokens = [line.split() for line in fileStream]
    dataPointsCount = len(tokens)-1
    dimensionality = 4
    X = [[0.0 for x in range(dimensionality)] for x in range(dataPointsCount)]
    Y = [0.0 for x in range(dataPointsCount)]
    i = 1
    j = 1
    while i < len(tokens):
        while j < dimensionality+2:
            if(j == dimensionality+1):
                Y[i-1] = int(tokens[i][j])
            else:
                X[i-1][j-1] = int(tokens[i][j])
            j = j + 1
        j = 0
        i = i + 1
    return X, Y

trainingFileName = sys.argv[1]
testingFileName = sys.argv[2]

trainingX, trainingY = parseFile(trainingFileName)
testingX, testingY = parseFile(testingFileName)

classifier = tree.DecisionTreeClassifier(criterion="entropy")
classifier = classifier.fit(trainingX, trainingY)

predictedY = classifier.predict(testingX)
i = 0
print("Number of movies = " + str(len(testingY)))
truePositives = 0
falsePositives = 0
trueNegatives = 0
falseNegatives = 0

while i<len(testingY):
    if testingY[i] == 1 :
        if testingY[i] == predictedY[i]:
            truePositives += 1
        else:
            falsePositives += 1
    else:
        if testingY[i] == predictedY[i]:
            trueNegatives += 1
        else:
            falseNegatives += 1
    i += 1

print("True positives = " + str(truePositives))
print("True negatives = " + str(trueNegatives))
print("False positives = " + str(falsePositives))
print("False Negatives = " + str(falseNegatives))
print("Error Rate = " + str((falsePositives + falseNegatives) / len(testingY)))