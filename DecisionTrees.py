import numpy as np
from sklearn import model_selection, preprocessing, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import ensemble
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import re
import html
import pandas as pd



def check_testSet(model, train_x, train_y, test_x):
    # fit the training dataset on the given model
    model.fit(train_x, train_y)

    # predict the labels on test dataset
    predictions = model.predict(test_x)

    return



def process_set(train_x, test_x):
    # vectorize text on a word level
    vectorizer = CountVectorizer()
    vectors_train = vectorizer.fit_transform(train_x)
    vectors_test = vectorizer.transform(test_x)
    # vectorize on  word level using tf-idf
    tf_idf_vectorizer = TfidfVectorizer(analyzer='word',stop_words="english")

    vectors_train_idf = tf_idf_vectorizer.fit_transform(train_x)
    vectors_test_idf = tf_idf_vectorizer.transform(test_x)

    # normalize
    vectors_train_idf_normalized = normalize(vectors_train_idf)
    vectors_test_idf_normalized = normalize(vectors_test_idf)


    return vectors_train_idf_normalized, vectors_test_idf_normalized

#filter everything except letters
def cleanHtml(t):
    t = html.unescape(t)
    filter = re.compile('[^a-zA-Z ]')
    t = filter.sub(' ',t)
    return t

#get set and cleans it if clean is set to True
def get_set(setFile,clean):
    # load dataset into a panda dataframe
    parsed_Set = pd.read_csv(setFile, header=0, sep=',')
    # print(parsed_Set['comments'][0:20])
    #cleanup text
    if clean:
        print("cleaning")
        parsed_Set['comments']=parsed_Set['comments'].apply(cleanHtml)


    # return
    train_x, test_x, train_y, test_y = model_selection.train_test_split(
        parsed_Set['comments'], parsed_Set['subreddits'], train_size=0.8, test_size=0.2)

    return train_x, test_x, train_y, test_y

def train_model(model, train_x, train_y, test_x, test_y):
    # fit the training dataset on the given model
    model.fit(train_x, train_y)

    # predict the labels on test dataset
    predictions = model.predict(test_x)
    return metrics.accuracy_score(predictions, test_y)

def gridsearch(model,tuned_parameters,scores, train_x, train_y, test_x, test_y):
    for score in scores:
        print("Gridsearch Started")

        clf = GridSearchCV(model, tuned_parameters, cv=5,scoring=score)
        clf.fit(train_x, train_y)

        print("Best parameters found")
        print(clf.best_params_)


def main():
    training_setfile = 'sets\\reddit_train.csv'
    testing_setfile = 'sets\\reddit_test.csv'

    # kaggle submission
    # parsed_Set = pd.read_csv(training_setfile, header=0, sep=',')
    # parsed_Test = pd.read_csv(testing_setfile, header=0, sep=',')
    # kaggle_train_x = parsed_Set['comments']
    # kaggle_train_y = parsed_Set['subreddits']
    # kaggle_test_x = parsed_Test['comments']
    #
    # kaggle_train_x, kaggle_test_x = process_set(kaggle_train_x, kaggle_test_x)
    # model_NBmulti = naive_bayes.MultinomialNB(alpha=0.19)
    # model_NBmulti.fit(kaggle_train_x, kaggle_train_y)
    # predictions = model_NBmulti.predict(kaggle_test_x)
    #
    # pd.DataFrame(predictions).to_csv("sets\\prediction.csv", header=["Category"], index_label="Id")
    # print("kaggle prediction saved")

    #testing
    scoring = ['accuracy']
    raw_set = get_set(training_setfile, True)
    train_x, test_x = process_set(raw_set[0], raw_set[1])
    train_y, test_y = raw_set[2], raw_set[3]

    # gridsearch multinomial naive bayes
    # tuned_parameters = [{'alpha': [(1.0/100)*x for x in range(1,100) ]}]
    # grid_NBmulti = naive_bayes.MultinomialNB()
    # gridsearch(grid_NBmulti,tuned_parameters,scoring, train_x, train_y, test_x, test_y)

    # test multinomial naive bayes
    model_NBmulti = naive_bayes.MultinomialNB(alpha=0.19)
    acc = train_model(model_NBmulti, train_x, train_y, test_x, test_y)
    print("naive bayes multi with idf: ", acc)

    # gridsearch linearSVC
    # tuned_parameters = [{'tol': [(1e-9/x) for x in range(1, 3)],'C':[0.19]}]
    # grid_SVC = svm.LinearSVC()
    # gridsearch(grid_SVC,tuned_parameters,scoring, train_x, train_y, test_x, test_y)

    # test linearSVC
    # model_SVC =svm.LinearSVC(C=0.19, tol=1e-10)
    # acc = train_model(model_SVC, train_x, train_y, test_x, test_y)
    # print("linear SVM with idf: ", acc)

    #gridsearch SGDC
    # tuned_parameters = [{'alpha': [(1e-4/x) for x in range(1, 5)], 'max_iter': [10,30,60]}]
    # grid_SGDC = SGDClassifier(tol=None)
    # gridsearch(grid_SGDC, tuned_parameters, scoring, train_x, train_y, test_x, test_y)

    #test SGDC
    # model_SGDC =SGDClassifier(loss='hinge', penalty='l2',alpha=3.3e-5, max_iter=30, tol=None)
    # acc = train_model(model_SGDC, train_x, train_y, test_x, test_y)
    # print("SGD with idf: ", acc)


    # model_logisticregression = LogisticRegression(solver='lbfgs', max_iter=500,multi_class='auto')
    # acc = train_model(model_logisticregression, train_x, train_y, test_x, test_y)
    # print("logistic regression with idf: ", acc)

    # model_ADA = ensemble.AdaBoostClassifier(n_estimators=50,learning_rate=10)
    # acc = train_model(model_ADA, train_x, train_y, test_x, test_y)
    # print("ADA with idf: ", acc)
    #
    # model_ADA = ensemble.AdaBoostClassifier(n_estimators=50, learning_rate=1)
    # acc = train_model(model_ADA, train_x, train_y, test_x, test_y)
    # print("ADA with idf: ", acc)
    #
    # model_ADA = ensemble.AdaBoostClassifier(algorithm='SAMME',n_estimators=50,
    #                                         base_estimator=SGDClassifier(loss='hinge'), learning_rate=1)
    # acc = train_model(model_ADA, train_x, train_y, test_x, test_y)
    # print("ADA tree with idf: ", acc)
    #


    return


if __name__ == '__main__':
    main()
