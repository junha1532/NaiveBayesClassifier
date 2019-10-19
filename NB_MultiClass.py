import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn import model_selection, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectPercentile
from sklearn.preprocessing import normalize, binarize


class naiveBayes:

    thetaYJ = []
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

        for i in range(len(self.predictionClass)):
            self.thetaY.append(classTS[i].shape[0] / countTS)

        numCols = trainX.shape[1]
        for i in range(len(self.predictionClass)):
            appendList = []
            for j in range(numCols):
                count = classTS[i][:, j].count_nonzero()
                appendList.append( (count + self.alpha) / (classTS[i].shape[0] + 2 * self.alpha) )
            self.thetaYJ.append(appendList)
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
            # featureLikelihood = 0
            # for j in range(numFeatures):
            #     if j in trainExample.indices:  # check if test sample's feature is 1
            #         featureLikelihood += np.log(self.thetaYJ[k][j])
            #     else:
            #         featureLikelihood += np.log(1 - self.thetaYJ[k][j])
            featureLikelihood = np.where(trainExample.todense() == 1, np.log(self.thetaYJ[k]), np.log(np.ones(len(self.thetaYJ[k])) - self.thetaYJ[k]))
            featureLikelihood = np.sum(featureLikelihood)
            cpList.append(featureLikelihood + np.log(self.thetaY[k]))
        return cpList

def main():

    # parse set and generate training and test sets
    training_setfile = 'sets\\reddit_train.csv'
    testing_setfile = 'sets\\reddit_test.csv'
    # load dataset into a panda dataframe
    parsed_Set = pd.read_csv(training_setfile, header=0, sep=',')
    parsed_Test = pd.read_csv(testing_setfile, header=0, sep=',')
    train_x, test_x, train_y, test_y = model_selection.train_test_split(
             parsed_Set['comments'], parsed_Set['subreddits'], train_size=0.8, test_size=0.2)

    # train_x = parsed_Set['comments']
    # train_y = parsed_Set['subreddits']
    # test_x = parsed_Test['comments']

    # tf*idf vectorizer
    vectorizer = TfidfVectorizer(binary= True, use_idf=False, norm=None)
    vectors_train_idf = vectorizer.fit_transform(train_x)
    vectors_test_idf = vectorizer.transform(test_x)
    # normalization
    #vectors_train_idf_normalized = normalize(vectors_train_idf)
    #vectors_test_idf_normalized = normalize(vectors_test_idf)
    selector = SelectPercentile(percentile=10)
    vectors_train_idf_selected = selector.fit_transform(vectors_train_idf, train_y)
    vectors_test_idf_selected = selector.transform(vectors_test_idf)

    nb = naiveBayes(alpha = 1.0)
    nb.fit(vectors_train_idf_selected, train_y)
    predictions = nb.predict(vectors_test_idf_selected)
    acc = metrics.accuracy_score(predictions, test_y)
    print("Naive Bayes prediction = ", acc)
    # pd.DataFrame(predictions).to_csv("sets\\prediction.csv", header = ["Category"], index_label= "Id")
    # print("done!")

    # scores = model_selection.cross_val_score(nb, vectors_train_idf, train_y, cv=5)
    # print("Naive Bayes prediction = ", sum(scores) / len(scores))

    
    

if __name__ == '__main__':
    main()
