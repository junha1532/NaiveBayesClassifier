import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
class naiveBayes:
    animeTS = None
    AskRedditTS = None
    baseballTS = None
    canadaTS = None
    conspiracyTS = None
    europeTS = None
    funnyTS = None
    gameofthronesTS = None
    GlobalOffensiveTS = None
    hockeyTS = None
    leagueoflegnedsTS = None
    moviesTS = None
    MusicTS = None
    nbaTS = None
    nflTS = None
    OverwatchTS = None
    soccerTS = None
    treesTS = None
    worldnewsTS = None
    wowTS = None

    countTS = None

    def __init__(self, trainSet):
        self.countTS = trainSet.shape[0] # number of training examples

        self.setTS(trainSet)

    def setTS(self, trainSet):
        self.animeTS = trainSet[np.where(trainSet[:, -1] == "anime")]
        self.AskRedditTS = trainSet[np.where(trainSet[:, -1] == "AskReddit")]
        self.baseballTS = trainSet[np.where(trainSet[:, -1] == "baseball")]
        self.canadaTS = trainSet[np.where(trainSet[:, -1] == "canada")]
        self.conspiracyTS = trainSet[np.where(trainSet[:, -1] == "conspiracy")]
        self.europeTS = trainSet[np.where(trainSet[:, -1] == "europe")]
        self.funnyTS = trainSet[np.where(trainSet[:, -1] == "funny")]
        self.gameofthronesTS = trainSet[np.where(trainSet[:, -1] == "gameofthrones")]
        self.GlobalOffensiveTS = trainSet[np.where(trainSet[:, -1] == "GlobalOffensive")]
        self.hockeyTS = trainSet[np.where(trainSet[:, -1] == "hockey")]
        self.leagueoflegnedsTS = trainSet[np.where(trainSet[:, -1] == "leagueoflegends")]
        self.moviesTS = trainSet[np.where(trainSet[:, -1] == "movies")]
        self.MusicTS = trainSet[np.where(trainSet[:, -1] == "Music")]
        self.nbaTS = trainSet[np.where(trainSet[:, -1] == "nba")]
        self.nflTS = trainSet[np.where(trainSet[:, -1] == "nfl")]
        self.OverwatchTS = trainSet[np.where(trainSet[:, -1] == "Overwatch")]
        self.soccerTS = trainSet[np.where(trainSet[:, -1] == "soccer")]
        self.treesTS = trainSet[np.where(trainSet[:, -1] == "trees")]
        self.worldnewsTS = trainSet[np.where(trainSet[:, -1] == "worldnews")]
        self.wowTS = trainSet[np.where(trainSet[:, -1] == "wow")]

    def predict(self, testSet):
        numRows = testSet.shape[0]
        predictList = []
        for i in range(numRows):
            x = testSet[i]
            numCols = x.shape[1] - 1
            # this is initial p(y=c)
            animeProb = self.animeTS.shape[0] / self.countTS
            AskRedditProb = self.AskRedditTS.shape[0] / self.countTS
            baseballProb = self.baseballTS.shape[0] / self.countTS
            canadaProb = self.canadaTS.shape[0] / self.countTS
            conspiracyProb = self.conspiracyTS.shape[0] / self.countTS
            europeProb = self.europeTS.shape[0] / self.countTS
            funnyProb = self.funnyTS.shape[0] / self.countTS
            gameofthronesProb = self.gameofthronesTS.shape[0] / self.countTS
            GlobalOffensiveProb = self.GlobalOffensiveTS.shape[0] / self.countTS
            hockeyProb = self.hockeyTS.shape[0] / self.countTS
            leagueoflegendsProb = self.leagueoflegnedsTS.shape[0] / self.countTS
            moviesProb = self.moviesTS.shape[0] / self.countTS
            MusicProb = self.MusicTS.shape[0] / self.countTS
            nbaProb = self.nbaTS.shape[0] / self.countTS
            nflProb = self.nflTS.shape[0] / self.countTS
            OverwatchProb = self.OverwatchTS.shape[0] / self.countTS
            soccerProb = self.soccerTS.shape[0] / self.countTS
            treesProb = self.treesTS.shape[0] / self.countTS
            worldnewsProb = self.worldnewsTS.shape[0] / self.countTS
            wowProb= self.wowTS.shape[0] / self.countTS

            cpList = []

            for j  in range(numCols):
                set = self.animeTS[np.where(self.animeTS[:,j] == 1)]
                animeProb = animeProb * ((set.shape[0] + 1) / (self.animeTS.shape[0] + 2))

                set = self.AskRedditTS[np.where(self.AskRedditTS[:, j] == 1)]
                AskRedditProb = AskRedditProb * ((set.shape[0] + 1) / (self.AskRedditTS.shape[0] + 2))

                set = self.baseballTS[np.where(self.baseballTS[:, j] == 1)]
                baseballProb = baseballProb * ((set.shape[0] + 1) / (self.baseballTS.shape[0] + 2))

                set = self.canadaTS[np.where(self.canadaTS[:, j] == 1)]
                canadaProb = canadaProb * ((set.shape[0] + 1) / (self.canadaTS.shape[0] + 2))

                set = self.conspiracyTS[np.where(self.conspiracyTS[:, j] == 1)]
                conspiracyProb = conspiracyProb * ((set.shape[0] + 1) / (self.conspiracyTS.shape[0] + 2))

                set = self.europeTS[np.where(self.europeTS[:, j] == 1)]
                europeProb = europeProb * ((set.shape[0] + 1) / (self.europeTS.shape[0] + 2))

                set = self.funnyTS[np.where(self.funnyTS[:, j] == 1)]
                funnyProb = funnyProb * ((set.shape[0] + 1) / (self.funnyTS.shape[0] + 2))

                set = self.gameofthronesTS[np.where(self.animeTS[:, j] == 1)]
                gameofthronesProb = gameofthronesProb * ((set.shape[0] + 1) / (self.gameofthronesTS.shape[0] + 2))

                set = self.GlobalOffensiveTS[np.where(self.GlobalOffensiveTS[:, j] == 1)]
                GlobalOffensiveProb = GlobalOffensiveProb * ((set.shape[0] + 1) / (self.GlobalOffensiveTS.shape[0] + 2))

                set = self.hockeyTS[np.where(self.animeTS[:, j] == 1)]
                hockeyProb = hockeyProb * ((set.shape[0] + 1) / (self.hockeyTS.shape[0] + 2))

                set = self.leagueoflegnedsTS[np.where(self.leagueoflegnedsTS[:, j] == 1)]
                leagueoflegendsProb = leagueoflegendsProb * ((set.shape[0] + 1) / (self.leagueoflegnedsTS.shape[0] + 2))

                set = self.moviesTS[np.where(self.moviesTS[:, j] == 1)]
                moviesProb = moviesProb * ((set.shape[0] + 1) / (self.moviesTS.shape[0] + 2))

                set = self.MusicTS[np.where(self.MusicTS[:, j] == 1)]
                MusicProb = MusicProb * ((set.shape[0] + 1) / (self.MusicTS.shape[0] + 2))

                set = self.nbaTS[np.where(self.nbaTS[:, j] == 1)]
                nbaProb = nbaProb * ((set.shape[0] + 1) / (self.nbaTS.shape[0] + 2))

                set = self.nflTS[np.where(self.nflTS[:, j] == 1)]
                nflProb = nflProb * ((set.shape[0] + 1) / (self.nflTS.shape[0] + 2))

                set = self.OverwatchTS[np.where(self.OverwatchTS[:, j] == 1)]
                OverwatchProb = OverwatchProb * ((set.shape[0] + 1) / (self.OverwatchTS.shape[0] + 2))

                set = self.soccerTS[np.where(self.soccerTS[:, j] == 1)]
                soccerProb = soccerProb * ((set.shape[0] + 1) / (self.soccerTS.shape[0] + 2))

                set = self.treesTS[np.where(self.treesTS[:, j] == 1)]
                treesProb = treesProb * ((set.shape[0] + 1) / (self.treesTS.shape[0] + 2))

                set = self.worldnewsTS[np.where(self.worldnewsTS[:, j] == 1)]
                worldnewsProb = worldnewsProb * ((set.shape[0] + 1) / (self.worldnewsTS.shape[0] + 2))

                set = self.wowTS[np.where(self.wowTS[:, j] == 1)]
                wowProb = wowProb * ((set.shape[0] + 1) / (self.wowTS.shape[0] + 2))

            cpList.append(animeProb)
            cpList.append(AskRedditProb)
            cpList.append(baseballProb)
            cpList.append(canadaProb)
            cpList.append(conspiracyProb)
            cpList.append(europeProb)
            cpList.append(funnyProb)
            cpList.append(gameofthronesProb)
            cpList.append(GlobalOffensiveProb)
            cpList.append(hockeyProb)
            cpList.append(leagueoflegendsProb)
            cpList.append(moviesProb)
            cpList.append(MusicProb)
            cpList.append(nbaProb)
            cpList.append(nflProb)
            cpList.append(OverwatchProb)
            cpList.append(soccerProb)
            cpList.append(treesProb)
            cpList.append(worldnewsProb)
            cpList.append(wowProb)

            maxProbClass = cpList.index(max(cpList))  # highest probability belonging to class index (eg. anime = 0, AskReddit = 1, ..)
            predictList.append(maxProbClass) # add the index of the highest probabilty class

        return predictList

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
