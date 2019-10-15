import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile
from sklearn.preprocessing import normalize


class naiveBayes:

    thetaJY = {}
    thetaY = []

    def __init__(self):
        pass

    def fit(self, trainX, trainY):

        countTS = trainX.shape[0]  # number of training examples
        askRedditTS = trainX[np.where(trainY[:] == "anime")]
        animeTS = trainX[np.where(trainY[:] == "AskReddit")]
        baseballTS = trainX[np.where(trainY[:] == "baseball")]
        canadaTS = trainX[np.where(trainY[:] == "canada")]
        conspiracyTS = trainX[np.where(trainY[:] == "conspiracy")]
        europeTS = trainX[np.where(trainY[:] == "europe")]
        funnyTS = trainX[np.where(trainY[:] == "funny")]
        gameofthronesTS = trainX[np.where(trainY[:] == "gameofthrones")]
        globalOffensiveTS = trainX[np.where(trainY[:] == "GlobalOffensive")]
        hockeyTS = trainX[np.where(trainY[:] == "hockey")]
        leagueoflegnedsTS = trainX[np.where(trainY[:] == "leagueoflegends")]
        moviesTS = trainX[np.where(trainY[:] == "movies")]
        musicTS = trainX[np.where(trainY[:] == "Music")]
        nbaTS = trainX[np.where(trainY[:] == "nba")]
        nflTS = trainX[np.where(trainY[:] == "nfl")]
        overwatchTS = trainX[np.where(trainY[:] == "Overwatch")]
        soccerTS = trainX[np.where(trainY[:] == "soccer")]
        treesTS = trainX[np.where(trainY[:] == "trees")]
        worldnewsTS = trainX[np.where(trainY[:] == "worldnews")]
        wowTS = trainX[np.where(trainY[:] == "wow")]

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

        numCols = trainX.shape[1]

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

    # parse set and generate training and test sets
    training_setfile = 'sets\\reddit_train.csv'
    testing_setfile = 'sets\\reddit_test.csv'
    # load dataset into a panda dataframe
    parsed_Set = pd.read_csv(training_setfile, header=0, sep=',')
    parsed_Test = pd.read_csv(testing_setfile, header=0, sep=',')
    # train_x, test_x, train_y, test_y = model_selection.train_test_split(
    #          parsed_Set['comments'], parsed_Set['subreddits'], train_size=0.8, test_size=0.2)

    train_x = parsed_Set['comments']
    train_y = parsed_Set['subreddits']
    test_x = parsed_Test['comments']

    # tf*idf vectorizer
    tf_idf_vectorizer = TfidfVectorizer(analyzer='word')
    vectors_train_idf = tf_idf_vectorizer.fit_transform(train_x)
    vectors_test_idf = tf_idf_vectorizer.transform(test_x)
    # normalization
    vectors_train_idf_normalized = normalize(vectors_train_idf)
    vectors_test_idf_normalized = normalize(vectors_test_idf)
    # pre-processing / selector
    # sel = VarianceThreshold(threshold=(0.8))
    # vectors_train_idf_selected = sel.fit(vectors_train_idf_normalized)
    # vectors_test_idf_selected = sel.fit(vectors_test_idf_normalized)
    selector = SelectPercentile(percentile=25)
    vectors_train_idf_selected = selector.fit_transform(vectors_train_idf_normalized, train_y)
    vectors_test_idf_selected = selector.transform(vectors_test_idf_normalized)

    nb = naiveBayes()
    nb.fit(vectors_train_idf_selected, train_y)
    predictions = nb.predict(vectors_test_idf_selected)
    pd.DataFrame(predictions).to_csv("sets\\prediction.csv", header = ["Category"], index_label= "Id")
    print("done!")

    
    

if __name__ == '__main__':
    main()
