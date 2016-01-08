# -*- coding: utf-8 -*-
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold


def main():
    l_encoder = pickle.load(open('encoder.pickle'))
    train = pd.read_csv('dataset/transformed_train.csv')
    test = pd.read_csv('dataset/transformed_test.csv')

    test['Vote'] = l_encoder.inverse_transform(test['Vote'])
    train['Vote'] = l_encoder.inverse_transform(train['Vote'])


    # print 'make 7 win:'
    # evaluate_manipulation(l_encoder, test, train, make_7_win)
    # print 'strengthen_coalition'
    # evaluate_manipulation(l_encoder, test, train, strengthen_coalition)
    print 'change_coalition:'
    evaluate_manipulation(l_encoder, test, train, change_coalition)


def evaluate_manipulation(l_encoder, test, train, manipulation):
    # Cross-validation on train set
    print 'Cross-validation on train set\n'
    counts = pd.DataFrame()
    # counts['real'] = train.Vote.value_counts()
    # counts['real'] = 0
    # counts['predicted'] = 0
    X = train.drop('Vote', axis=1)
    Y = train.Vote.values
    n_folds = 5
    kf = KFold(n=len(train), n_folds=n_folds, shuffle=True)
    for k, (train_index, test_index) in enumerate(kf):
        x_test = X.loc[test_index]

        classifier = DecisionTreeClassifier(max_depth=10)
        classifier.fit(X.values[train_index], Y[train_index])

        proba_before = pd.DataFrame(classifier.predict_proba(x_test.values))
        proba_before.columns = l_encoder.classes_

        manipulation(x_test)

        proba = pd.DataFrame(classifier.predict_proba(x_test.values))
        proba.columns = l_encoder.classes_

        if k==0:
            counts['prediction'] = proba_before.sum()
            counts['manipulated'] = proba.sum()
            counts['real'] = train.loc[test_index].Vote.value_counts()
        else:
            counts['prediction'] += proba_before.sum()
            counts['manipulated'] += proba.sum()
            counts['real'] += train.loc[test_index].Vote.value_counts()
    counts['%vote'] = counts.prediction / len(train) * 100
    counts['%manipulated_vote'] = counts.manipulated / len(train) * 100
    # counts['%change'] = (counts.manipulated - counts.real) * 100 / counts.real
    counts['manipulated'] /= n_folds
    counts['prediction'] /= n_folds
    counts['real'] /= n_folds
    print_df(counts)
    # train Vs test
    print '\n\ntrain Vs test\n'
    classifier = DecisionTreeClassifier(max_depth=10)
    counts = pd.DataFrame()
    classifier.fit(train.drop('Vote', axis=1).values, train.Vote.values)
    proba_before = pd.DataFrame(classifier.predict_proba(test.drop('Vote', axis=1)))
    proba_before.columns = l_encoder.classes_
    manipulation(test)
    proba = pd.DataFrame(classifier.predict_proba(test.drop('Vote', axis=1)))
    proba.columns = l_encoder.classes_
    counts['prediction'] = proba_before.sum()
    counts['manipulated'] = proba.sum()
    counts['real'] = test.Vote.value_counts()
    counts['%vote'] = counts.prediction / len(test) * 100
    counts['%manipulated_vote'] = counts.manipulated / len(test) * 100
    # counts['%change'] = (counts.manipulated - counts.real) * 100 / counts.real
    print_df(counts)


def print_df(df):
    print df[['real', 'prediction', '%vote', 'manipulated', '%manipulated_vote']]


def make_7_win(test):
    test['Overall_happiness_score'] += 1.267
    test['Yearly_IncomeK'] += 0.79
    test['Yearly_ExpensesK'] -= 0.487


def strengthen_coalition(test):
    test['Will_vote_only_large_party'] = 1
    test['Financial_agenda_matters'] = 0


def change_coalition(test):
    # test['Yearly_ExpensesK'] = +1
    # test['Yearly_IncomeK'] = +1
    # test['Overall_happiness_score'] = +1
    test['Financial_agenda_matters'] = 1


if __name__ == "__main__":
    main()
