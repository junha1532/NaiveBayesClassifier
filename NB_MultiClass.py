import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectPercentile
from sklearn.preprocessing import normalize, binarize


class naiveBayes:

    thetaJY = {}
    thetaY = []
    predictionClass = {}

    def __init__(self, alpha = 1.0, binarize = 0.0):
        self.alpha = alpha
        self.binarize = binarize

    def fit(self, trainX, trainY):

        if self.binarize is not None:
            trainX = binarize(trainX, threshold = self.binarize)

        classNames = np.unique(trainY)
        for i in range(len(classNames)):
            self.predictionClass[i] = classNames[i]

        countTS = trainX.shape[0]  # number of training examples

        classTS = [] #Training set divided for each class in belongs to
        for i in range(len(self.predictionClass)):
            classTS.append(trainX[np.where(trainY[:] == self.predictionClass[i])])
        # animeTS = trainX[np.where(trainY[:] == self.predictionClass[0])]
        # askRedditTS = trainX[np.where(trainY[:] == self.predictionClass[1])]
        # baseballTS = trainX[np.where(trainY[:] == self.predictionClass[2])]
        # canadaTS = trainX[np.where(trainY[:] == self.predictionClass[3])]
        # conspiracyTS = trainX[np.where(trainY[:] == self.predictionClass[4])]
        # europeTS = trainX[np.where(trainY[:] == self.predictionClass[5])]
        # funnyTS = trainX[np.where(trainY[:] == self.predictionClass[6])]
        # gameofthronesTS = trainX[np.where(trainY[:] == self.predictionClass[7])]
        # globalOffensiveTS = trainX[np.where(trainY[:] == self.predictionClass[8])]
        # hockeyTS = trainX[np.where(trainY[:] == self.predictionClass[9])]
        # leagueoflegnedsTS = trainX[np.where(trainY[:] == self.predictionClass[10])]
        # moviesTS = trainX[np.where(trainY[:] == self.predictionClass[11])]
        # musicTS = trainX[np.where(trainY[:] == self.predictionClass[12])]
        # nbaTS = trainX[np.where(trainY[:] == self.predictionClass[13])]
        # nflTS = trainX[np.where(trainY[:] == self.predictionClass[14])]
        # overwatchTS = trainX[np.where(trainY[:] == self.predictionClass[15])]
        # soccerTS = trainX[np.where(trainY[:] == self.predictionClass[16])]
        # treesTS = trainX[np.where(trainY[:] == self.predictionClass[17])]
        # worldnewsTS = trainX[np.where(trainY[:] == self.predictionClass[18])]
        # wowTS = trainX[np.where(trainY[:] == self.predictionClass[19])]

        for i in range(len(self.predictionClass)):
            self.thetaY.append(classTS[i].shape[0] / countTS)
        # self.thetaY.append(animeTS.shape[0] / countTS)
        # self.thetaY.append(askRedditTS.shape[0] / countTS)
        # self.thetaY.append(baseballTS.shape[0] / countTS)
        # self.thetaY.append(canadaTS.shape[0] / countTS)
        # self.thetaY.append(conspiracyTS.shape[0] / countTS)
        # self.thetaY.append(europeTS.shape[0] / countTS)
        # self.thetaY.append(funnyTS.shape[0] / countTS)
        # self.thetaY.append(gameofthronesTS.shape[0] / countTS)
        # self.thetaY.append(globalOffensiveTS.shape[0] / countTS)
        # self.thetaY.append(hockeyTS.shape[0] / countTS)
        # self.thetaY.append(leagueoflegnedsTS.shape[0] / countTS)
        # self.thetaY.append(moviesTS.shape[0] / countTS)
        # self.thetaY.append(musicTS.shape[0] / countTS)
        # self.thetaY.append(nbaTS.shape[0] / countTS)
        # self.thetaY.append(nflTS.shape[0] / countTS)
        # self.thetaY.append(overwatchTS.shape[0] / countTS)
        # self.thetaY.append(soccerTS.shape[0] / countTS)
        # self.thetaY.append(treesTS.shape[0] / countTS)
        # self.thetaY.append(worldnewsTS.shape[0] / countTS)
        # self.thetaY.append(wowTS.shape[0] / countTS)

        numCols = trainX.shape[1]

        for j in range(numCols):
            self.thetaJY[j] = {}

            for i in range(len(self.predictionClass)):
                count = classTS[i][:, j].count_nonzero()
                self.thetaJY[j][i] = (count + self.alpha) / (classTS[i].shape[0] + 2 * self.alpha)

            # count = animeTS[:, j].count_nonzero()
            # self.thetaJY[j][0] = (count + self.alpha) / (animeTS.shape[0] + 2*self.alpha)
            #
            # count = askRedditTS[:, j].count_nonzero()
            # self.thetaJY[j][1] = (count + self.alpha) / (askRedditTS.shape[0] + 2*self.alpha)
            #
            # count = baseballTS[:, j].count_nonzero()
            # self.thetaJY[j][2] = (count + self.alpha) / (baseballTS.shape[0] + 2*self.alpha)
            #
            # count = canadaTS[:, j].count_nonzero()
            # self.thetaJY[j][3] = (count + self.alpha) / (canadaTS.shape[0] + 2*self.alpha)
            #
            # count = conspiracyTS[:, j].count_nonzero()
            # self.thetaJY[j][4] = (count + self.alpha) / (conspiracyTS.shape[0] + 2*self.alpha)
            #
            # count = europeTS[:, j].count_nonzero()
            # self.thetaJY[j][5] = (count + self.alpha) / (europeTS.shape[0] + 2*self.alpha)
            #
            # count = funnyTS[:, j].count_nonzero()
            # self.thetaJY[j][6] = (count + self.alpha) / (funnyTS.shape[0] + 2*self.alpha)
            #
            # count = gameofthronesTS[:, j].count_nonzero()
            # self.thetaJY[j][7] = (count + self.alpha) / (gameofthronesTS.shape[0] + 2*self.alpha)
            #
            # count = globalOffensiveTS[:, j].count_nonzero()
            # self.thetaJY[j][8] = (count + self.alpha) / (globalOffensiveTS.shape[0] + 2*self.alpha)
            #
            # count = hockeyTS[:, j].count_nonzero()
            # self.thetaJY[j][9] = (count + self.alpha) / (hockeyTS.shape[0] + 2*self.alpha)
            #
            # count = leagueoflegnedsTS[:, j].count_nonzero()
            # self.thetaJY[j][10] = (count + self.alpha) / (leagueoflegnedsTS.shape[0] + 2*self.alpha)
            #
            # count = moviesTS[:, j].count_nonzero()
            # self.thetaJY[j][11] = (count + self.alpha) / (moviesTS.shape[0] + 2*self.alpha)
            #
            # count = musicTS[:, j].count_nonzero()
            # self.thetaJY[j][12] = (count + self.alpha) / (musicTS.shape[0] + 2*self.alpha)
            #
            # count = nbaTS[:, j].count_nonzero()
            # self.thetaJY[j][13] = (count + self.alpha) / (nbaTS.shape[0] + 2*self.alpha)
            #
            # count = nflTS[:, j].count_nonzero()
            # self.thetaJY[j][14] = (count + self.alpha) / (nflTS.shape[0] + 2*self.alpha)
            #
            # count = overwatchTS[:, j].count_nonzero()
            # self.thetaJY[j][15] = (count + self.alpha) / (overwatchTS.shape[0] + 2*self.alpha)
            #
            # count = soccerTS[:, j].count_nonzero()
            # self.thetaJY[j][16] = (count + self.alpha) / (soccerTS.shape[0] + 2*self.alpha)
            #
            # count = treesTS[:, j].count_nonzero()
            # self.thetaJY[j][17] = (count + self.alpha) / (treesTS.shape[0] + 2*self.alpha)
            #
            # count = worldnewsTS[:, j].count_nonzero()
            # self.thetaJY[j][18] = (count + self.alpha) / (worldnewsTS.shape[0] + 2*self.alpha)
            #
            # count = wowTS[:, j].count_nonzero()
            # self.thetaJY[j][19] = (count + self.alpha) / (wowTS.shape[0] + 2*self.alpha)

    def predict(self, testSet):
        numRows = testSet.shape[0] #number of testset
        numClasses = len(self.predictionClass)
        numFeatures = testSet.shape[1]
        predictList = [] #collection of prediction class for each test example
        for i in range(numRows):
            x = testSet[i]
            predictList.append(self.class_prob_list(numClasses, numFeatures, x))
        agmxList = np.argmax(predictList, axis = 1) #list of indices of the max value in each list within predictList
        # this is for returning a list of class names instead of indices
        return [self.predictionClass[x] for x in agmxList]

    def class_prob_list(self, numClasses, numFeatures, trainExample):
        cpList = []  # class probability list (prediction probability that comment could be from the specific community)
        for k in range(numClasses):
            featureLikelihood = 0
            for j in range(numFeatures):
                if j in trainExample.indices:  # check if test sample's feature is 1
                    featureLikelihood += np.log(self.thetaJY[j][k])
                else:
                    featureLikelihood += np.log(1 - self.thetaJY[j][k])
            cpList.append(featureLikelihood + np.log(self.thetaY[k]))
        return cpList

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
    selector = SelectPercentile(percentile=0.01)
    vectors_train_idf_selected = selector.fit_transform(vectors_train_idf, train_y)
    vectors_test_idf_selected = selector.transform(vectors_test_idf)

    nb = naiveBayes(alpha = 0.1)
    nb.fit(vectors_train_idf_selected, train_y)
    predictions = nb.predict(vectors_test_idf_selected)
    pd.DataFrame(predictions).to_csv("sets\\prediction.csv", header = ["Category"], index_label= "Id")
    print("done!")

    
    

if __name__ == '__main__':
    main()
