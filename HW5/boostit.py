import sys
import numpy as np
#from sklearn import metrics


def read_file(filename):
    return np.loadtxt(filename, skiprows=1)

class simpleLinearClassifier:
    w = None
    t = None
    def __init__(self, weights, P, N): # finds w and t the parameters of the classifier
        posCentroid = np.sum([p * w for w, p in zip(P, weights[:len(P)])], axis=0) / np.sum(weights[:len(P)])
        negCentroid = np.sum([n * w for w, n in zip(N, weights[len(P):])], axis=0) / np.sum(weights[len(P):])
        self.w = posCentroid - negCentroid
        self.t = (posCentroid + negCentroid) / 2

    def eval(self, x): # computes margin/perpendicular distance
        return (np.dot(x - self.t, self.w)) / np.linalg.norm(self.w)

    def error(self, P, N, weights):
        errors = 0
        for i in range(len(weights)):
            if i < len(P):
                if self.eval(P[i]) <= 0:
                    errors += weights[i]
            else:
                if self.eval(N[i - len(P)]) > 0:
                    errors += weights[i]

        return errors / (np.sum(weights))




def createSimpleLinearClassifier(weights, P, N):
    return simpleLinearClassifier(weights, P, N)

def boosting(P, N, T, A = createSimpleLinearClassifier):
    weights = [1 for _ in range(len(P) + len(N))]
    alphas = []
    weakClassifiers = []
    for t in range(T):
        newClassifier = A(weights, P, N)
        e = newClassifier.error(P, N, weights)
        if e >= 0.5:
            T = t
            break
        alpha = np.log((1 - e) / e) / 2
        weakClassifiers.append(newClassifier)
        alphas.append(alpha)

        #adjust weights
        for i in range(len(weights)):
            if i < len(P):
                if newClassifier.eval(P[i]) > 0:
                    weights[i] = weights[i] / (2 *  (1 - e))
                else:
                    weights[i] = weights[i] / (2 * e)
            else:
                if newClassifier.eval(N[i - len(P)]) <= 0:
                    weights[i] = weights[i] / (2 *  (1 - e))
                else:
                    weights[i] = weights[i] / (2 * e)

        #PrintingCode
        print("Iteration " + str(t+1) + ":")
        print("Error = " + str(e))
        print("Alpha = " + str(alpha))
        print("Factor to increase weights = " + str(1/(2*e)))
        print("Factor to decrease weights = " + str(1 / (2 * (1-e))))
        #end PrintingCode
    return lambda x: 1 if np.sum([ a * M.eval(x) for M, a in zip(weakClassifiers, alphas) ]) > 0 else 0







# Main code starts here
modelCount = int(sys.argv[1])
trainPositive = read_file(sys.argv[2])    # Read training file name from the script parameters
trainNegative = read_file(sys.argv[3])    # Read testing file name from the script parameters
testPositive = read_file(sys.argv[4])
testNegative = read_file(sys.argv[5])

strongClassifier = boosting(trainPositive, trainNegative, modelCount)


falsePositives = 0
falseNegatives = 0
for p in testPositive:
    if strongClassifier(p) == 0:
        falseNegatives += 1
for n in testNegative:
    if strongClassifier(n) == 1:
        falsePositives += 1
print("Testing:")
print("False Positives: " + str(falsePositives))
print("False Negatives: " + str(falseNegatives))
print("Error rate: " + str(100 * (falseNegatives+falsePositives)/(len(testPositive)+len(testNegative))) + "%")