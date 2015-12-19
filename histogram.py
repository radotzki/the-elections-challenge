# -*- coding: utf-8 -*-
from collections import defaultdict
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
import theElectionChallenge


def main():
    l_encoder = pickle.load(open('encoder.pickle'))
    train = pd.read_csv('dataset/transformed_train.csv')
    test = pd.read_csv('dataset/transformed_test.csv')
    test['Vote'] = l_encoder.inverse_transform(test['Vote'])
    counts = pd.DataFrame()
    counts['real'] = test.Vote.value_counts()
    # test_predictions = pd.read_csv('dataset/test_predictions.csv')



    #using predict_proba

    for name, classifier in theElectionChallenge.classifiers.iteritems():
        print name
        classifier.fit(train.drop('Vote', axis=1), train.Vote.values)
        print 'Division of voters by label prediction:'
        counts['predicted'] = pd.Series(l_encoder.inverse_transform(classifier.predict(test.drop('Vote', axis=1)))).value_counts()
        counts['difference'] = counts.real - counts.predicted
        # print counts
        print 'Total error: ' + str(counts.difference.abs().sum())

        if hasattr(classifier, 'predict_proba'):
            print 'Division of voters by probabilistic prediction:'
            proba = pd.DataFrame(classifier.predict_proba(test.drop('Vote', axis=1)))
            proba.columns = l_encoder.classes_
            counts['predicted'] = proba.sum()
            counts['difference'] = counts.real - counts.predicted
            # print counts
            print 'Total error: ' + str(counts.difference.abs().sum())


if __name__ == "__main__":
    main()
