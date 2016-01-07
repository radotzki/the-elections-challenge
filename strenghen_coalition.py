# -*- coding: utf-8 -*-
from collections import defaultdict
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
import modeling


def main():
    l_encoder = pickle.load(open('encoder.pickle'))
    train = pd.read_csv('dataset/transformed_train.csv')
    test = pd.read_csv('dataset/transformed_test.csv')
    test['Will_vote_only_large_party'] = 1
    test['Financial_agenda_matters'] = 0

    test['Vote'] = l_encoder.inverse_transform(test['Vote'])
    counts = pd.DataFrame()
    # counts['real'] = test.Vote.value_counts()

    classifier= DecisionTreeClassifier(max_depth=10)
    classifier.fit(train.drop('Vote', axis=1), train.Vote.values)

    print 'Division of voters by probabilistic prediction:'
    proba = pd.DataFrame(classifier.predict_proba(test.drop('Vote', axis=1)))
    proba.columns = l_encoder.classes_
    counts['predicted'] = proba.sum()
    # counts['difference'] = counts.real - counts.predicted
    print counts
    print 'Total coalition votes: ' + str(counts['predicted']['Blues'] + counts['predicted']['Greys']
                                          + counts['predicted']['Oranges']+ counts['predicted']['Reds']
                                          + counts['predicted']['Yellows'])

if __name__ == "__main__":
    main()
