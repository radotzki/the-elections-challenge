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

    # Cross-validation on train set
    print 'Cross-validation on train set\n'
    counts = pd.DataFrame()
    counts['real'] = train.Vote.value_counts()
    counts['real'] = 0
    counts['predicted'] = 0
    X = train.drop('Vote', axis=1)
    Y = train.Vote.values
    n_folds = 5
    kf = KFold(n=len(train), n_folds=n_folds, shuffle=True)

    for k, (train_index, test_index) in enumerate(kf):
        x_test = X.loc[test_index]
        counts['real'] += train.loc[test_index].Vote.value_counts()

        classifier = DecisionTreeClassifier(max_depth=10)
        classifier.fit(X.values[train_index], Y[train_index])

        x_test['Overall_happiness_score'] += 1.267
        x_test['Yearly_IncomeK'] += 0.79
        x_test['Yearly_ExpensesK'] -= 0.487

        proba = pd.DataFrame(classifier.predict_proba(x_test.values))
        proba.columns = l_encoder.classes_
        counts['predicted'] += proba.sum()

    counts['% vote'] = counts.predicted / len(train) * 100
    counts['% change'] = (counts.predicted - counts.real) * 100 / counts.real
    counts['predicted'] /= n_folds
    counts['real'] /= n_folds
    print counts


    # train Vs test
    print '\n\ntrain Vs test\n'
    classifier = DecisionTreeClassifier(max_depth=10)
    counts = pd.DataFrame()
    counts['real'] = test.Vote.value_counts()

    classifier.fit(train.drop('Vote', axis=1).values, train.Vote.values)

    test['Overall_happiness_score'] += 1.267
    test['Yearly_IncomeK'] += 0.79
    test['Yearly_ExpensesK'] -= 0.487

    proba = pd.DataFrame(classifier.predict_proba(test.drop('Vote', axis=1)))
    proba.columns = l_encoder.classes_
    counts['predicted'] = proba.sum()
    counts['% vote'] = counts.predicted / len(test) * 100
    counts['% change'] = (counts.predicted - counts.real) * 100 / counts.real
    print counts

if __name__ == "__main__":
    main()
