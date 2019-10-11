import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
class naiveBayes:

    thetaJY = {}
    thetaY = []

    def __init__(self):
        pass

    def fit(self, trainSet):

        countTS = trainSet.shape[0]  # number of training examples

        animeTS = trainSet[np.where(trainSet[:, -1] == "anime")]
        askRedditTS = trainSet[np.where(trainSet[:, -1] == "AskReddit")]
        baseballTS = trainSet[np.where(trainSet[:, -1] == "baseball")]
        canadaTS = trainSet[np.where(trainSet[:, -1] == "canada")]
        conspiracyTS = trainSet[np.where(trainSet[:, -1] == "conspiracy")]
        europeTS = trainSet[np.where(trainSet[:, -1] == "europe")]
        funnyTS = trainSet[np.where(trainSet[:, -1] == "funny")]
        gameofthronesTS = trainSet[np.where(trainSet[:, -1] == "gameofthrones")]
        globalOffensiveTS = trainSet[np.where(trainSet[:, -1] == "GlobalOffensive")]
        hockeyTS = trainSet[np.where(trainSet[:, -1] == "hockey")]
        leagueoflegnedsTS = trainSet[np.where(trainSet[:, -1] == "leagueoflegends")]
        moviesTS = trainSet[np.where(trainSet[:, -1] == "movies")]
        musicTS = trainSet[np.where(trainSet[:, -1] == "Music")]
        nbaTS = trainSet[np.where(trainSet[:, -1] == "nba")]
        nflTS = trainSet[np.where(trainSet[:, -1] == "nfl")]
        overwatchTS = trainSet[np.where(trainSet[:, -1] == "Overwatch")]
        soccerTS = trainSet[np.where(trainSet[:, -1] == "soccer")]
        treesTS = trainSet[np.where(trainSet[:, -1] == "trees")]
        worldnewsTS = trainSet[np.where(trainSet[:, -1] == "worldnews")]
        wowTS = trainSet[np.where(trainSet[:, -1] == "wow")]

        self.thetaY[0] = animeTS.shape[0] / countTS
        self.thetaY[1] = askRedditTS.shape[0] / countTS
        self.thetaY[2] = baseballTS.shape[0] / countTS
        self.thetaY[3] = canadaTS.shape[0] / countTS
        self.thetaY[4] = conspiracyTS.shape[0] / countTS
        self.thetaY[5] = europeTS.shape[0] / countTS
        self.thetaY[6] = funnyTS.shape[0] / countTS
        self.thetaY[7] = gameofthronesTS.shape[0] / countTS
        self.thetaY[8] = globalOffensiveTS.shape[0] / countTS
        self.thetaY[9] = hockeyTS.shape[0] / countTS
        self.thetaY[10] = leagueoflegnedsTS.shape[0] / countTS
        self.thetaY[11] = moviesTS.shape[0] / countTS
        self.thetaY[12] = musicTS.shape[0] / countTS
        self.thetaY[13] = nbaTS.shape[0] / countTS
        self.thetaY[14] = nflTS.shape[0] / countTS
        self.thetaY[15] = overwatchTS.shape[0] / countTS
        self.thetaY[16] = soccerTS.shape[0] / countTS
        self.thetaY[17] = treesTS.shape[0] / countTS
        self.thetaY[18] = worldnewsTS.shape[0] / countTS
        self.thetaY[19] = wowTS.shape[0] / countTS

        numCols = trainSet.shape[1] -1

        for j in range(numCols):
            self.thetaJY[j] = {}

            set = animeTS[np.where(animeTS[:, j] == 1)]
            self.thetaJY[j][0] = (set.shape[0] + 1) / (animeTS.shape[0] + 2)

            set = askRedditTS[np.where(askRedditTS[:, j] == 1)]
            self.thetaJY[j][1] = (set.shape[0] + 1) / (askRedditTS.shape[0] + 2)

            set = baseballTS[np.where(baseballTS[:, j] == 1)]
            self.thetaJY[j][2] = (set.shape[0] + 1) / (baseballTS.shape[0] + 2)

            set = canadaTS[np.where(canadaTS[:, j] == 1)]
            self.thetaJY[j][3] = (set.shape[0] + 1) / (canadaTS.shape[0] + 2)

            set = conspiracyTS[np.where(conspiracyTS[:, j] == 1)]
            self.thetaJY[j][4] = (set.shape[0] + 1) / (conspiracyTS.shape[0] + 2)

            set = europeTS[np.where(europeTS[:, j] == 1)]
            self.thetaJY[j][5] = (set.shape[0] + 1) / (europeTS.shape[0] + 2)

            set = funnyTS[np.where(funnyTS[:, j] == 1)]
            self.thetaJY[j][6] = (set.shape[0] + 1) / (funnyTS.shape[0] + 2)

            set = gameofthronesTS[np.where(gameofthronesTS[:, j] == 1)]
            self.thetaJY[j][7] = (set.shape[0] + 1) / (gameofthronesTS.shape[0] + 2)

            set = globalOffensiveTS[np.where(globalOffensiveTS[:, j] == 1)]
            self.thetaJY[j][8] = (set.shape[0] + 1) / (globalOffensiveTS.shape[0] + 2)

            set = hockeyTS[np.where(hockeyTS[:, j] == 1)]
            self.thetaJY[j][9] = (set.shape[0] + 1) / (hockeyTS.shape[0] + 2)

            set = leagueoflegnedsTS[np.where(leagueoflegnedsTS[:, j] == 1)]
            self.thetaJY[j][10] = (set.shape[0] + 1) / (leagueoflegnedsTS.shape[0] + 2)

            set = moviesTS[np.where(moviesTS[:, j] == 1)]
            self.thetaJY[j][11] = (set.shape[0] + 1) / (moviesTS.shape[0] + 2)

            set = musicTS[np.where(musicTS[:, j] == 1)]
            self.thetaJY[j][12] = (set.shape[0] + 1) / (musicTS.shape[0] + 2)

            set = nbaTS[np.where(nbaTS[:, j] == 1)]
            self.thetaJY[j][13] = (set.shape[0] + 1) / (nbaTS.shape[0] + 2)

            set = nflTS[np.where(nflTS[:, j] == 1)]
            self.thetaJY[j][14] = (set.shape[0] + 1) / (nflTS.shape[0] + 2)

            set = overwatchTS[np.where(overwatchTS[:, j] == 1)]
            self.thetaJY[j][15] = (set.shape[0] + 1) / (overwatchTS.shape[0] + 2)

            set = soccerTS[np.where(soccerTS[:, j] == 1)]
            self.thetaJY[j][16] = (set.shape[0] + 1) / (soccerTS.shape[0] + 2)

            set = treesTS[np.where(treesTS[:, j] == 1)]
            self.thetaJY[j][17] = (set.shape[0] + 1) / (treesTS.shape[0] + 2)

            set = worldnewsTS[np.where(askRedditTS[:, j] == 1)]
            self.thetaJY[j][18] = (set.shape[0] + 1) / (askRedditTS.shape[0] + 2)

            set = wowTS[np.where(wowTS[:, j] == 1)]
            self.thetaJY[j][19] = (set.shape[0] + 1) / (wowTS.shape[0] + 2)



    def predict(self, testSet):
        numRows = testSet.shape[0] #number of testset
        numClasses = 20
        numFeatures = testSet.shape[1]
        predictList = [] #collection of prediction class for each test example
        for i in range(numRows):
            x = testSet[i]
            cpList = [] # class probability list (prediction probability that comment could be from the specific community)
            for k in range(numClasses):
                featureLikelihood = 0
                for j in range(numFeatures):
                    featureLikelihood += x[j] * np.log(self.thetaJY[k][j]) + (1 - x[j]) * np.log(1 - self.thetaJY[k][j])
                cpList[k] = featureLikelihood + np.log(self.thetaY[k])
            predictList[i] = cpList.index(max(cpList))

        # this is for converting to numpy (column) array/vector
        return np.asarray(predictList).transpose()

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
