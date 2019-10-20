# NaiveBayesClassifier

*ModelAccuracy.py contains accuracy tests for all classification models with different preprocessing with features
When you run the ModelAccuracy.py, it will give the 5-fold accuracies of different models and the average of the 5-fold accuracies

*NB_MultiClass.py contains Bernoulli Naive Bayes model implemented from scratch and all the preprocessing with features.
When you run the MultiClass.py, it will give the 5-fold accuracies of the reddit_train data set.
**NOTE**: I have implemented the matrix operation instead of for loops in predict() method to decrease the run-time. However, the model is still quite slow
Takes Around 25mins to run the 5-fold