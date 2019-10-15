import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectPercentile
from sklearn.preprocessing import normalize


class naiveBayes:

    thetaJY = {}
    thetaY = []

    predictionClass = {
        0:"anime",
        1:"AskReddit",
        2:"baseball",
        3:"canada",
        4:"conspiracy",
        5:"europe",
        6:"funny",
        7:"gameofthrones",
        8:"GlobalOffensive",
        9:"hockey",
        10:"leagueoflegends",
        11:"movies",
        12:"Music",
        13:"nba",
        14:"nfl",
        15:"Overwatch",
        16:"soccer",
        17:"trees",
        18:"worldnews",
        19:"wow"
    }

    def __init__(self):
        pass

    def fit(self, trainX, trainY):

        countTS = trainX.shape[0]  # number of training examples
        animeTS = trainX[np.where(trainY[:] == self.predictionClass[0])]
        #a = animeTS[0][6006]

        askRedditTS = trainX[np.where(trainY[:] == self.predictionClass[1])]
        baseballTS = trainX[np.where(trainY[:] == self.predictionClass[2])]
        canadaTS = trainX[np.where(trainY[:] == self.predictionClass[3])]
        conspiracyTS = trainX[np.where(trainY[:] == self.predictionClass[4])]
        europeTS = trainX[np.where(trainY[:] == self.predictionClass[5])]
        funnyTS = trainX[np.where(trainY[:] == self.predictionClass[6])]
        gameofthronesTS = trainX[np.where(trainY[:] == self.predictionClass[7])]
        globalOffensiveTS = trainX[np.where(trainY[:] == self.predictionClass[8])]
        hockeyTS = trainX[np.where(trainY[:] == self.predictionClass[9])]
        leagueoflegnedsTS = trainX[np.where(trainY[:] == self.predictionClass[10])]
        moviesTS = trainX[np.where(trainY[:] == self.predictionClass[11])]
        musicTS = trainX[np.where(trainY[:] == self.predictionClass[12])]
        nbaTS = trainX[np.where(trainY[:] == self.predictionClass[13])]
        nflTS = trainX[np.where(trainY[:] == self.predictionClass[14])]
        overwatchTS = trainX[np.where(trainY[:] == self.predictionClass[15])]
        soccerTS = trainX[np.where(trainY[:] == self.predictionClass[16])]
        treesTS = trainX[np.where(trainY[:] == self.predictionClass[17])]
        worldnewsTS = trainX[np.where(trainY[:] == self.predictionClass[18])]
        wowTS = trainX[np.where(trainY[:] == self.predictionClass[19])]

        self.thetaY.append(animeTS.shape[0] / countTS)
        self.thetaY.append(askRedditTS.shape[0] / countTS)
        self.thetaY.append(baseballTS.shape[0] / countTS)
        self.thetaY.append(canadaTS.shape[0] / countTS)
        self.thetaY.append(conspiracyTS.shape[0] / countTS)
        self.thetaY.append(europeTS.shape[0] / countTS)
        self.thetaY.append(funnyTS.shape[0] / countTS)
        self.thetaY.append(gameofthronesTS.shape[0] / countTS)
        self.thetaY.append(globalOffensiveTS.shape[0] / countTS)
        self.thetaY.append(hockeyTS.shape[0] / countTS)
        self.thetaY.append(leagueoflegnedsTS.shape[0] / countTS)
        self.thetaY.append(moviesTS.shape[0] / countTS)
        self.thetaY.append(musicTS.shape[0] / countTS)
        self.thetaY.append(nbaTS.shape[0] / countTS)
        self.thetaY.append(nflTS.shape[0] / countTS)
        self.thetaY.append(overwatchTS.shape[0] / countTS)
        self.thetaY.append(soccerTS.shape[0] / countTS)
        self.thetaY.append(treesTS.shape[0] / countTS)
        self.thetaY.append(worldnewsTS.shape[0] / countTS)
        self.thetaY.append(wowTS.shape[0] / countTS)

        numCols = trainX.shape[1]

        for j in range(numCols):
            self.thetaJY[j] = {}

            count = animeTS[:, j].count_nonzero()
            self.thetaJY[j][0] = (count + 1) / (animeTS.shape[0] + 2)

            count = askRedditTS[:, j].count_nonzero()
            self.thetaJY[j][1] = (count + 1) / (askRedditTS.shape[0] + 2)

            count = baseballTS[:, j].count_nonzero()
            self.thetaJY[j][2] = (count + 1) / (baseballTS.shape[0] + 2)

            count = canadaTS[:, j].count_nonzero()
            self.thetaJY[j][3] = (count + 1) / (canadaTS.shape[0] + 2)

            count = conspiracyTS[:, j].count_nonzero()
            self.thetaJY[j][4] = (count + 1) / (conspiracyTS.shape[0] + 2)

            count = europeTS[:, j].count_nonzero()
            self.thetaJY[j][5] = (count + 1) / (europeTS.shape[0] + 2)

            count = funnyTS[:, j].count_nonzero()
            self.thetaJY[j][6] = (count + 1) / (funnyTS.shape[0] + 2)

            count = gameofthronesTS[:, j].count_nonzero()
            self.thetaJY[j][7] = (count + 1) / (gameofthronesTS.shape[0] + 2)

            count = globalOffensiveTS[:, j].count_nonzero()
            self.thetaJY[j][8] = (count + 1) / (globalOffensiveTS.shape[0] + 2)

            count = hockeyTS[:, j].count_nonzero()
            self.thetaJY[j][9] = (count + 1) / (hockeyTS.shape[0] + 2)

            count = leagueoflegnedsTS[:, j].count_nonzero()
            self.thetaJY[j][10] = (count + 1) / (leagueoflegnedsTS.shape[0] + 2)

            count = moviesTS[:, j].count_nonzero()
            self.thetaJY[j][11] = (count + 1) / (moviesTS.shape[0] + 2)

            count = musicTS[:, j].count_nonzero()
            self.thetaJY[j][12] = (count + 1) / (musicTS.shape[0] + 2)

            count = nbaTS[:, j].count_nonzero()
            self.thetaJY[j][13] = (count + 1) / (nbaTS.shape[0] + 2)

            count = nflTS[:, j].count_nonzero()
            self.thetaJY[j][14] = (count + 1) / (nflTS.shape[0] + 2)

            count = overwatchTS[:, j].count_nonzero()
            self.thetaJY[j][15] = (count + 1) / (overwatchTS.shape[0] + 2)

            count = soccerTS[:, j].count_nonzero()
            self.thetaJY[j][16] = (count + 1) / (soccerTS.shape[0] + 2)

            count = treesTS[:, j].count_nonzero()
            self.thetaJY[j][17] = (count + 1) / (treesTS.shape[0] + 2)

            count = worldnewsTS[:, j].count_nonzero()
            self.thetaJY[j][18] = (count + 1) / (worldnewsTS.shape[0] + 2)

            count = wowTS[:, j].count_nonzero()
            self.thetaJY[j][19] = (count + 1) / (wowTS.shape[0] + 2)

    def predict(self, testSet):
        numRows = testSet.shape[0] #number of testset
        numClasses = len(self.predictionClass)
        numFeatures = testSet.shape[1]
        predictList = [] #collection of prediction class for each test example
        for i in range(numRows):
            x = testSet[i]
            cpList = [] # class probability list (prediction probability that comment could be from the specific community)
            for k in range(numClasses):
                featureLikelihood = 0
                for j in range(numFeatures):
                    if i in x.indices: #check if test sample's feature is 1
                        featureLikelihood += np.log(self.thetaJY[j][k])
                    else:
                        featureLikelihood += np.log(1 - self.thetaJY[j][k])
                cpList.append(featureLikelihood + np.log(self.thetaY[k]))
            #add the right prediction
            index = cpList.index(max(cpList))
            predictList.append(self.predictionClass[index])
        # this is for converting to numpy (column) array/vector
        return np.array(predictList)

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
    vectorizer = CountVectorizer(binary = True)
    vectors_train_idf = vectorizer.fit_transform(train_x)
    vectors_test_idf = vectorizer.transform(test_x)
    # normalization
    #vectors_train_idf_normalized = normalize(vectors_train_idf)
    #vectors_test_idf_normalized = normalize(vectors_test_idf)
    selector = SelectPercentile(percentile=10)
    vectors_train_idf_selected = selector.fit_transform(vectors_train_idf, train_y)
    vectors_test_idf_selected = selector.transform(vectors_test_idf)

    nb = naiveBayes()
    nb.fit(vectors_train_idf_selected, train_y)
    predictions = nb.predict(vectors_test_idf_selected)
    pd.DataFrame(predictions).to_csv("sets\\prediction.csv", header = ["Category"], index_label= "Id")
    print("done!")

    
    

if __name__ == '__main__':
    main()
