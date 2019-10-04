import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
class naiveBayes:
    classOneTS = None
    classZeroTS = None

    probZero = None #p(y=0)
    probOne = None #p(y=1)

    cpZero = [] #p(xj = 1| y=0)
    cpOne = [] #p(xj = 1| y=1)

    countTS = None
    def __init__(self, trainSet):
        self.countTS = trainSet.shape[0] # number of training examples

        self.classOneTS = trainSet[np.where(trainSet[:, -1] == 1)]
        self.classZeroTS = trainSet[np.where(trainSet[:, -1] == 0)]
        self.calculateProb(self.countTS)  # Calculate distribution of class y=1 & class y=0 training examples
        self.calculateCP() # Calculate conditional Probability of feature = 1 (true) given y = 1 or 0

    def calculateProb(self, numRows):
        self.probOne = self.classOneTS.shape[0] / numRows
        self.probZero = self.classZeroTS.shape[0] / numRows

    def calculateCP(self):
        #for number of columns(attributes) calculate the number of training example where y =1 or y=0
        numCols = self.classOneTS.shape[1] - 1
        for i in range(numCols):
            set = self.classOneTS[np.where(self.classOneTS[:, i] == 1)]
            self.cpOne.append((set.shape[0] + 1) / (self.classOneTS.shape[0] + 2))  # add the number of class y = 1 with concerning attribute = 1
        numCols = self.classZeroTS.shape[1] - 1
        for i in range(numCols):
            set = self.classZeroTS[np.where(self.classZeroTS[:, i] == 1)]
            self.cpZero.append((set.shape[0] + 1) / (self.classOneTS.shape[0] + 2))  # add the number of class y = 0 with concerning attribute = 1

    def loRatioZeroFeature(self, j):
        return np.log((1-self.cpOne[j])/(1-self.cpZero[j]))

    def loRatioOneFeature(self, j):
        return np.log((self.cpOne[j])/(self.cpZero[j]))

    def loRatio(self, x):
        logRatio = np.log(self.probOne/self.probZero)

        for i in range(self.countTS):
            logRatio = logRatio + self.loRatioZeroFeature(i) + (self.loRatioOneFeature(i) - self.loRatioZeroFeature(i)) * x[i]

        return logRatio

    def predict(self, x):
        ratio = self.loRatio(x)
        if ratio > 0:
            return 1
        else:
            return 0


def main():

    reader = csv.reader(open('sets\\reddit_train.csv', "rt", encoding="utf8"))
    trainSet = list(reader) #create list of list of data
    trainSet = np.array(trainSet, dtype= object) #change datatype to numpy array with pointer to the original list (strings are too long to be stored!)

    trainSet = np.delete(trainSet, 0, 0) # delete first row
    trainSet = np.delete(trainSet, 0, 1) # delete first column

    print(trainSet[0][0])
    print(trainSet[0][1])
    #'sets\\reddit_train.csv'



if __name__ == '__main__':
    main()