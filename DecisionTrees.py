import numpy as np
from sklearn import model_selection, preprocessing, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import pandas as pd


def train_model(model, train_x, train_y, test_x, test_y):
    # fit the training dataset on the given model
    model.fit(train_x, train_y)

    # predict the labels on test dataset
    predictions = model.predict(test_x)

    return metrics.accuracy_score(predictions, test_y)


def process_set(train_x, test_x, train_y, test_y):
    # vectorize text on a word level
    vectorizer = CountVectorizer()
    vectors_train = vectorizer.fit_transform(train_x)
    vectors_test = vectorizer.transform(test_x)
    # vectorize on  word level using tf-idf
    tf_idf_vectorizer = TfidfVectorizer(analyzer='word')
    vectors_train_idf = tf_idf_vectorizer.fit_transform(train_x)
    vectors_test_idf = tf_idf_vectorizer.transform(test_x)
    # normalize
    vectors_train_normalized = normalize(vectors_train)
    vectors_test_normalized = normalize(vectors_test)
    vectors_train_idf_normalized = normalize(vectors_train_idf)
    vectors_test_idf_normalized = normalize(vectors_test_idf)
    # label one hot encoding
    binarizer = preprocessing.LabelBinarizer()
    train_y_binarized = binarizer.fit_transform(train_y)
    test_y_binarized = binarizer.fit_transform(test_y)
    # shape label array into a matrix of correct format
    train_y_binarized = np.argmax(train_y_binarized, axis=1)
    test_y_binarized = np.argmax(test_y_binarized, axis=1)

    return vectors_train_idf_normalized, vectors_test_idf_normalized, train_y, test_y


def get_set(setFile):
    # load dataset into a panda dataframe
    parsed_Set = pd.read_csv(setFile, header=0, sep=',')
    train_x, test_x, train_y, test_y = model_selection.train_test_split(
        parsed_Set['comments'], parsed_Set['subreddits'], train_size=0.8, test_size=0.2)

    return train_x, test_x, train_y, test_y


def main():
    # parse set and generate training and test sets
    training_setfile = 'sets\\reddit_train.csv'
    raw_set = get_set(training_setfile)
    train_x, test_x, train_y, test_y = process_set(raw_set[0], raw_set[1], raw_set[2], raw_set[3])
    # initialize models with parameters
    model_decisiontree = DecisionTreeClassifier(criterion='gini', max_depth=300)
    model_naivebayes = naive_bayes.MultinomialNB()
    model_logisticregression = LogisticRegression()
    model_linearsvc = LinearSVC()
    # train and test the models for accuracy
    acc = train_model(model_naivebayes, train_x, train_y, test_x, test_y)
    print("naive bayes with idf: ", acc)
    acc = train_model(model_logisticregression, train_x, train_y, test_x, test_y)
    print("logistic regression with idf: ", acc)
    acc = train_model(model_decisiontree, train_x, train_y, test_x, test_y)
    print("Decision Tree with idf: ", acc)
    acc = train_model(model_linearsvc, train_x, train_y, test_x, test_y)
    print("LinearSVC with idf: ", acc)


if __name__ == '__main__':
    main()
