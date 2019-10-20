import numpy as np
import pandas as pd
from nltk import word_tokenize
from sklearn import model_selection, preprocessing, naive_bayes, metrics, svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, chi2, mutual_info_classif, SelectFromModel
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF
# from nltk import word_tokenize
# from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
stop_words = set(stopwords.words('english'))


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

# kf = KFold(n_splits=5) # Define the split - into 2 folds
# kf.get_n_splits(train_x)
# for train_index, test_index in kf.split(train_x):
#     x_train, x_test = train_x[train_index], train_x[test_index]
#     y_train, y_test = train_y[train_index], train_y[test_index]


#tf*idf vectorizer
tf_idf_vectorizer = TfidfVectorizer(analyzer='word')
vectors_train_idf = tf_idf_vectorizer.fit_transform(train_x)
vectors_test_idf = tf_idf_vectorizer.transform(test_x)
# #CounVectorizer
# vectorizer = CountVectorizer()
# vectors_train_idf = vectorizer.fit_transform(train_x)
# vectors_test_idf = vectorizer.transform(test_x)
#normalization
vectors_train_idf_normalized = normalize(vectors_train_idf)
vectors_test_idf_normalized = normalize(vectors_test_idf)
#pre-processing / selector
selector = SelectPercentile(percentile=10)
vectors_train_idf_selected = selector.fit_transform(vectors_train_idf_normalized, train_y)
vectors_test_idf_selected = selector.transform(vectors_test_idf_normalized)

# tuned_parameters = [{'alpha': [0.19, 1.0]}]
# n_folds = 5
# nb = GridSearchCV(nb, tuned_parameters, cv=n_folds, refit=False)
# nb.fit(vectors_train_idf_selected,train_y)
# scores = nb.cv_results_["mean_test_score"]
# print("scores = ",scores)

nb = naive_bayes.MultinomialNB()
scores = model_selection.cross_val_score(nb, vectors_train_idf_selected, train_y, cv=5)
print("Naive Bayes prediction = ", scores, sum(scores)/5)

grid_SVC = svm.LinearSVC()
scores = model_selection.cross_val_score(grid_SVC, vectors_train_idf_selected, train_y, cv=5)
print("LinearSVC prediction = ", scores, sum(scores)/5)

model_logisticregression = LogisticRegression()
scores = model_selection.cross_val_score(model_logisticregression, vectors_train_idf_selected, train_y, cv=5)
print("Logistic regression prediction = ", scores, sum(scores)/5)

sgd = SGDClassifier()
scores = model_selection.cross_val_score(sgd, vectors_train_idf_selected, train_y, cv=5)
print("SGD prediction = ", scores, sum(scores)/5)

tree = DecisionTreeClassifier()
scores = model_selection.cross_val_score(tree, vectors_train_idf_selected, train_y, cv=5)
print("Tree prediction = ", scores, sum(scores)/5)

