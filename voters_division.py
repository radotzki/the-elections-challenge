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
    test['Vote'] = l_encoder.inverse_transform(test['Vote'])
    counts = pd.DataFrame()
    counts['real'] = test.Vote.value_counts()
    best_prediction_error = len(test)  # max possible value
    scores = modeling.cross_validation(train, modeling.CLASSIFIERS)
    classifiers = modeling.CLASSIFIERS.copy()
    classifiers.update({'My classifier': modeling.MyClassifier(scores),
                      'My classifier 2': modeling.MyClassifier2(scores)})

    for name, classifier in classifiers.iteritems():
        print name
        classifier.fit(train.drop('Vote', axis=1), train.Vote.values)
        print 'Division of voters by label prediction:'
        counts['predicted'] = pd.Series(l_encoder.inverse_transform(classifier.predict(test.drop('Vote', axis=1)))).value_counts()
        counts['difference'] = counts.real - counts.predicted
        print counts
        total_error = counts.difference.abs().sum()/2
        if total_error<best_prediction_error:
            best_prediction_error=total_error
            best_prediction = counts.copy(True)
        print 'Total error: ' + str(total_error)

        if hasattr(classifier, 'predict_proba'):
            print 'Division of voters by probabilistic prediction:'
            proba = pd.DataFrame(classifier.predict_proba(test.drop('Vote', axis=1)))
            proba.columns = l_encoder.classes_
            counts['predicted'] = proba.sum()
            counts['difference'] = counts.real - counts.predicted
            print counts
            total_error = counts.difference.abs().sum()/2
            if total_error<best_prediction_error:
                best_prediction_error=total_error
                best_prediction = counts.copy(deep=True)
            print 'Total error: ' + str(total_error)

        print '-------------------------'
        print ''

    print "Best prediction for division of voters:"
    print best_prediction


if __name__ == "__main__":
    main()
