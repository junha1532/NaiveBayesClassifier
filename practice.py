import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing, naive_bayes, metrics, svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, chi2, mutual_info_classif, SelectFromModel
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF
# from nltk import word_tokenize
# from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
#
stop_words = set(stopwords.words('english'))
# tokenized_stop_words = word_tokenize(' '.join(stopwords.words('english')))
# class Tokenizer(object):
#     def __init__(self):
#         self.stemmer = PorterStemmer()
#
#     def _stem(self, token):
#         if (token in stop_words):
#             return token  # Solves error "UserWarning: Your stop_words may be inconsistent with your preprocessing."
#         return self.stemmer.stem(token)
#
#     def __call__(self, line):
#         tokens = word_tokenize(line)
#         tokens = (self._stem(token) for token in tokens)  # Stemming
#         return list(tokens)

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
# sel = VarianceThreshold(threshold=(0.8))
# vectors_train_idf_selected = sel.fit(vectors_train_idf_normalized)
# vectors_test_idf_selected = sel.fit(vectors_test_idf_normalized)
selector = SelectPercentile(percentile=25)
vectors_train_idf_selected = selector.fit_transform(vectors_train_idf_normalized, train_y)
vectors_test_idf_selected = selector.transform(vectors_test_idf_normalized)

# tuned_parameters = [{'alpha': [0.19, 0.3, 0.4, 0.05, 0.5, 0.1]}]
# n_folds = 5

nb = naive_bayes.MultinomialNB(alpha = 0.19)
# nb = GridSearchCV(nb, tuned_parameters, cv=n_folds, refit=False)


nb.fit(vectors_train_idf_selected, train_y)
predictions = nb.predict(vectors_test_idf_selected)
# pd.DataFrame(predictions).to_csv("sets\\prediction.csv", header = ["Category"], index_label= "Id")
# print("done!")

acc = metrics.accuracy_score(predictions, test_y)
print("Naive Bayes prediction = ", acc)

# scores = nb.cv_results_["mean_test_score"]
# print("scores = ",scores)
