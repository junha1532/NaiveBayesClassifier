import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from nltk import word_tokenize
from scipy.sparse import csr_matrix, vstack
from sklearn import model_selection, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectPercentile
from sklearn.preprocessing import normalize, binarize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
stop_words = set(stopwords.words('english'))


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
def kfold(folds, model, train_x, train_y):
    foldSize = train_x.shape[0] // folds
    fold_accuracy = []
    for i in range(folds):
        start = i * foldSize
        end = (i + 1) * foldSize - 1
        testSubset = train_x[start:end]
        yTestSubset = train_y[start:end]
        if i == folds - 1:
            trainSubset = train_x[0:start]
            yTrainSubset = train_y[0:start]
        elif i ==0:
            trainSubset = train_x[end:]
            yTrainSubset = train_y[end:]
        else:
            trainSubset = vstack((train_x[0:start], train_x[end:]))
            yTrainSubset = pd.concat([train_y[0:start], train_y[end:]], ignore_index=True)

        model.fit(trainSubset, yTrainSubset)

        predictions = model.predict(testSubset)
        acc = metrics.accuracy_score(predictions, yTestSubset)
        # acc = evaluate_acc(model, testSubset, yTestSubset)
        fold_accuracy.append(acc)
    return fold_accuracy
# def evaluate_acc(model, testSet, y):
#     accuracy = 0
#     numRows = testSet.shape[0]
#     for i in range(numRows):
#         yi = y[i]
#         x = np.asmatrix(testSet[i]).transpose()
#         prediction = model.predict(x)
#         if prediction == yi:
#             accuracy += 1.0
#     return accuracy / numRows

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

    nb = naiveBayes(alpha=1.0)
    kfold_avg = kfold(5, nb, vectors_train_idf_selected, train_y)
    print("Model k-fold accuracies : ", kfold_avg)

    # nb.fit(vectors_train_idf_selected, train_y)
    # predictions = nb.predict(vectors_test_idf_selected)
    # acc = metrics.accuracy_score(predictions, test_y)
    # print("naive bayes multi with idf: ", acc)

    # pd.DataFrame(predictions).to_csv("sets\\prediction.csv", header = ["Category"], index_label= "Id")
    # print("done!")

    # scores = model_selection.cross_val_score(nb, vectors_train_idf, train_y, cv=5)
    # print("Naive Bayes prediction = ", sum(scores) / len(scores))

    
    

if __name__ == '__main__':
    main()
