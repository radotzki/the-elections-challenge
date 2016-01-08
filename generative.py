__author__ = 'User'

from sklearn.cross_validation import KFold
from sklearn.naive_bayes import GaussianNB
import numpy
import numpy as np
import pandas as pd

def binary_transformation(df):
    df['Will_vote_only_large_party'] = df['Will_vote_only_large_party'].map(lambda x: 1 if x==1 else -1)
    df['Financial_agenda_matters'] = df['Financial_agenda_matters'].map(lambda x: 1 if x==1 else -1)


def dummis_transformation(df):
    dummies_df = pd.get_dummies(df['Most_Important_Issue'], prefix='Most_Important_Issue')
    df = pd.concat([df, dummies_df], axis=1, join='inner')
    return df.drop('Most_Important_Issue', axis=1)


def print_vector(v):
    for i in xrange(len(v)):
        print("%.2f\t" % v[i]),
    print ''


def main():
    df_train = pd.read_csv('./dataset/transformed_train.csv')
    df_test = pd.read_csv('./dataset/transformed_test.csv')
    # df = df_train
    binary_transformation(df_train)
    binary_transformation(df_test)
    df_train = dummis_transformation(df_train)
    df_test = dummis_transformation(df_test)

    clf = GaussianNB()
    clf.fit(df_train.drop('Vote', axis=1).values, df_train.Vote.values)

    for party in xrange(10):
        print 'party ' + str(party)
        print 'mean',
        print_vector(clf.theta_[party])
        print 'var ',
        print_vector(clf.sigma_[party])

    pass

if __name__ == "__main__":
    main()
