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


def most_important_features(df):
    alpha = 0.27
    X, y = df.drop('Vote', axis=1).values, df.Vote.values
    clf = DecisionTreeClassifier(max_depth=10)
    clf = clf.fit(X, y)
    important_features = clf.feature_importances_.tolist()
    important_features = [df.columns[important_features.index(f)] for f in important_features if f > alpha]
    return important_features


def print_vote_count(df, first_party, second_party):
    print 'First party votes = ' + str(len(df[df.Vote == first_party]))
    print 'Second party votes = ' + str(len(df[df.Vote == second_party]))
    print 'Diff = ' + str(len(df[df.Vote == first_party]) - len(df[df.Vote == second_party]))
    print '\n'


def get_two_largest_parties(df):
    parties = {}
    for party in df.Vote.unique():
        parties[party] = len(df[df.Vote == party])

    first_party = max(parties.iteritems(), key=operator.itemgetter(1))[0]
    del parties[first_party]
    second_party = max(parties.iteritems(), key=operator.itemgetter(1))[0]
    return [first_party, second_party]


def plot_one_party_vs_all(df, party, party_name):
    one_party_vs_all = df.copy()

    for index, row in one_party_vs_all.iterrows():
        if row.Vote != party:
            one_party_vs_all.loc[index, 'Vote'] = party + 1

    plot_density_by_most_important_features(one_party_vs_all, [party + 1, party], str(party_name) + ' vs all: ')


def plot_density_by_most_important_features(df, votes, title):
    for c in most_important_features(df):
        for vote in votes:
            df[c][df.Vote == vote].plot(kind='kde', title=title + c)
        plt.show()


def main():
    train = pd.read_csv('dataset/transformed_train.csv')
    test = pd.read_csv('dataset/transformed_test.csv')

    first_party, second_party = get_two_largest_parties(pd.concat([train, test]))

    print 'ground truth:'
    print_vote_count(pd.concat([train, test]), first_party, second_party)

    print 'with prediction:'
    print_vote_count(pd.concat([train, prediction(train, test)]), first_party, second_party)

    new_train = pd.concat([train[train.Vote == first_party], train[train.Vote == second_party]])
    features_to_manipulate = most_important_features(new_train)

    plot_density_by_most_important_features(new_train, [first_party, second_party], 'first party vs second party: ')
    plot_one_party_vs_all(train, first_party, 'first party')
    plot_one_party_vs_all(train, second_party, 'second party')

    # test['Yearly_ExpensesK'] = 0.5

    # Those manipulations will change the winning party
    # test.loc[test.Vote == first_party, features_to_manipulate[0]] = 0.6
    # test.loc[test.Vote == first_party, features_to_manipulate[1]] = 0
    # test.loc[test.Vote == second_party, features_to_manipulate[1]] = -0.71

    print 'with prediction and manipulation:'
    print_vote_count(pd.concat([train, prediction(train, test)]), first_party, second_party)

    print 'feature to manipulate: ' + str(features_to_manipulate)

if __name__ == "__main__":
    main()

