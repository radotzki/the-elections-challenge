# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import operator
from matplotlib import pyplot as plt


def prediction(train, test):
    label = 'Vote'
    classifier = DecisionTreeClassifier(max_depth=10)
    features = list(train.columns)
    features.remove(label)
    classifier.fit(train[list(features)], train[label].values)
    pred = test.copy()
    pred[label] = classifier.predict(test[list(features)])
    return pred


def most_important_features(train):
    alpha = 0.3
    X, y = train.drop('Vote', axis=1).values, train.Vote.values
    clf = DecisionTreeClassifier(max_depth=10)
    clf = clf.fit(X, y)
    important_features = clf.feature_importances_.tolist()
    important_features = [train.columns[important_features.index(f)] for f in important_features if f > alpha]
    return important_features


def print_vote_count(df, largest_party, second_large_party):
    print 'largest party votes = ' + str(len(df[df.Vote == largest_party]))
    print 'second large party votes = ' + str(len(df[df.Vote == second_large_party]))
    print 'diff = ' + str(len(df[df.Vote == largest_party]) - len(df[df.Vote == second_large_party]))
    print '\n'


def get_two_largest_parties(df):
    dict = {}
    for party in df.Vote.unique():
        dict[party] = len(df[df.Vote == party])

    largest_party = max(dict.iteritems(), key=operator.itemgetter(1))[0]
    del dict[largest_party]
    second_large_party = max(dict.iteritems(), key=operator.itemgetter(1))[0]
    return [largest_party, second_large_party]

def main():
    train = pd.read_csv('dataset/transformed_train.csv')
    test = pd.read_csv('dataset/transformed_test.csv')

    largest_party, second_large_party = get_two_largest_parties(pd.concat([train, test]))

    ########
    # large_vs_all = train.copy()
    # second_large_vs_all = train.copy()
    #
    # for index, row in train.iterrows():
    #     if row.Vote != largest_party:
    #         large_vs_all.loc[index, 'Vote'] = second_large_party
    #
    # print most_important_feature(large_vs_all)
    #
    # for c in [most_important_feature(large_vs_all)]:
    #     for vote in [largest_party, second_large_party]:
    #         large_vs_all[large_vs_all.Vote == vote][c].plot(kind='kde')
    #     plt.show()
    ########

    print 'ground truth:'
    print_vote_count(pd.concat([train, test]), largest_party, second_large_party)

    print 'with prediction:'
    print_vote_count(pd.concat([train, prediction(train, test)]), largest_party, second_large_party)

    new_train = pd.concat([train[train.Vote == largest_party], train[train.Vote == second_large_party]])
    features_to_manipulate = most_important_features(new_train)

    for c in features_to_manipulate:
        for vote in [largest_party, second_large_party]:
            new_train[new_train.Vote==vote][c].plot(kind='kde')
        plt.show()

    test.loc[test.Vote == largest_party, features_to_manipulate[0]] = 0.8
    test.loc[test.Vote == largest_party, features_to_manipulate[1]] = 0
    # test.loc[test.Vote == second_large_party, features_to_manipulate[0]] = -0.5
    test.loc[test.Vote == second_large_party, features_to_manipulate[1]] = -0.71

    print 'with prediction and manipulation:'
    print_vote_count(pd.concat([train, prediction(train, test)]), largest_party, second_large_party)

    print 'feature to manipulate: ' + str(features_to_manipulate)

if __name__ == "__main__":
    main()

