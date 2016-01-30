# -*- coding: utf-8 -*-

###############################################################################
################## Data preparation ###########################################
###############################################################################
from collections import defaultdict
import math
import numpy as np
from numpy.linalg import lstsq
import pickle
import operator
from sklearn.mixture import GMM
from sklearn.preprocessing import Imputer, MinMaxScaler
from scipy.stats import norm, kstest
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
import pandas as pd
import scipy
import scipy.interpolate
import random
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
CLASSIFIERS_GENERATORS = {
    "Random Forest 5": lambda: RandomForestClassifier(n_estimators=5),
    "Random Forest 10": lambda: RandomForestClassifier(n_estimators=10),
    "Random Forest 20": lambda: RandomForestClassifier(n_estimators=20),
    # "Random Forest 40": lambda: RandomForestClassifier(n_estimators=50),
    # "Random Forest 80": lambda: RandomForestClassifier(n_estimators=80),
    # "Random Forest 100": lambda: RandomForestClassifier(n_estimators=100),
    # "Random Forest 120": lambda: RandomForestClassifier(n_estimators=120),
    # "Decision Tree 5": lambda: DecisionTreeClassifier(max_depth=5),
    "Decision Tree 10": lambda: DecisionTreeClassifier(max_depth=10),
    "Decision Tree 20": lambda: DecisionTreeClassifier(max_depth=20),
    "Nearest Neighbors 5": lambda: KNeighborsClassifier(n_neighbors=5),
    # "Nearest Neighbors 21": lambda: KNeighborsClassifier(n_neighbors=21),
    # "Nearest Neighbors 26": lambda: KNeighborsClassifier(n_neighbors=26),
    # "Nearest Neighbors 35": lambda: KNeighborsClassifier(n_neighbors=35),
    "Nearest Neighbors 51": lambda: KNeighborsClassifier(n_neighbors=51),
    "Nearest Neighbors 7": lambda: KNeighborsClassifier(n_neighbors=7),
    # "AdaBoost 100": lambda: AdaBoostClassifier(DecisionTreeClassifier(max_depth=15), n_estimators=100),
    # "AdaBoost 500": lambda: AdaBoostClassifier(DecisionTreeClassifier(max_depth=15), n_estimators=500),
    # "AdaBoost 1000": lambda: AdaBoostClassifier(DecisionTreeClassifier(max_depth=15), n_estimators=1000),
    # "AdaBoost 1200": lambda: AdaBoostClassifier(DecisionTreeClassifier(max_depth=15), n_estimators=1200),
    # "AdaBoost12 1000": lambda: AdaBoostClassifier(DecisionTreeClassifier(max_depth=12), n_estimators=1000),
    # "rbf SVM OVO 100": lambda: SVC(kernel="rbf", C=100, probability=True),
    # "rbf SVM OVO 120": lambda: SVC(kernel="rbf", C=120, probability=True),
    # "rbf SVM OVO 500": lambda: SVC(kernel="rbf", C=500, probability=True),
    # "rbf SVM OVO 700": lambda: SVC(kernel="rbf", C=700, probability=True),
    # "rbf SVM OVO 900": lambda: SVC(kernel="rbf", C=900, probability=True),
    # "rbf SVM OVO 1100": lambda: SVC(kernel="rbf", C=1100, probability=True),
    # "GaussianNB": lambda: GaussianNB(),
}
CLASSIFIERS = {name: clf_gen() for name, clf_gen in CLASSIFIERS_GENERATORS.iteritems()}

CLUSTERS = {
    "GMM 11": GMM(n_components=11, covariance_type='diag'),
    "GMM 12": GMM(n_components=12, covariance_type='diag'),
    "GMM 13": GMM(n_components=13, covariance_type='diag'),
    "GMM 14": GMM(n_components=14, covariance_type='diag'),
    "GMM 15": GMM(n_components=15, covariance_type='diag'),
    "GMM 16": GMM(n_components=16, covariance_type='diag'),
}

def array_diff(a, b):
    b = set(b)
    return [aa for aa in a if aa not in b]


def split_features_by_type(_df):
    _df = _df.dropna()
    all_features = [c for c in _df.columns if c != 'Vote']
    all_features.remove("split")
    discrete_features = [c for c in all_features if len(_df[c].unique()) <= 20]
    continuous_features = array_diff(all_features, discrete_features)
    categorical_features = list(_df.keys()[_df.dtypes.map(lambda x: x == 'object')])
    categorical_features.remove("Vote")
    binary_features = [c for c in all_features if len(_df[c].unique()) == 2]
    numeric_features = array_diff(all_features, categorical_features) + binary_features
    return [all_features, discrete_features, continuous_features, categorical_features, numeric_features]


def mark_negative_values_as_nan(_df):
    positive_features = [
        'Avg_monthly_expense_when_under_age_21',
        'AVG_lottary_expanses',
        'Yearly_ExpensesK',
        'Yearly_IncomeK',
        'Avg_monthly_expense_on_pets_or_plants',
        'Avg_monthly_household_cost',
        'Phone_minutes_10_years',
        'Avg_size_per_room',
        'Garden_sqr_meter_per_person_in_residancy_area'
    ]
    for f in [x for x in _df.columns if x in positive_features]:
        _df[f] = _df[f].map(lambda x: x if x >= 0 else np.nan)
    return _df


def outlier_detection(_df, features):
    """
    for all continuous features: keep only values that are within +3 to -3 standard deviations, otherwise set nan
    """
    for f in features:
        std = _df[f].std()
        mean = _df[f].mean()
        _df[f] = _df[f].map(lambda x: x if np.abs(x - mean) <= (3 * std) else np.nan)
    return _df


def find_coefficients(Xs, Ys, exponents):
    X = tuple((tuple((pow(x, p) for p in exponents)) for x in Xs))
    y = tuple(((y) for y in Ys))
    x, resids, rank, s = lstsq(X, y)
    return x


def fill_f1_by_f2_linear(df, f1, f2):
    rows_to_complete = df[f1].isnull() & df[f2].notnull()

    df_dropna = df[[f1, f2]].dropna()
    coefs = find_coefficients(df_dropna[f2], df_dropna[f1], range(2))  # linear approximation
    # for i, row in df.iterrows():
    #     if rows_to_complete[i]:
    #         x=df[f1][i]
    #         print x
    #         print y_interp(x)
    #         df[f2][i] = y_interp(df[f1][i])
    df[f1][rows_to_complete] = df[f2][rows_to_complete].map(lambda x: coefs[0] + coefs[1] * x)


def label_encoder(df):
    le = preprocessing.LabelEncoder()
    le.fit(df.Vote.values)
    df.Vote = le.transform(df.Vote.values)
    return le


def label_decoder(df, le):
    return le.inverse_transform(df.Vote.values)


def categorical_features_transformation(_df):
    # Identify which of the original features are objects
    obj_feat = _df.keys()[_df.dtypes.map(lambda x: x == 'object')]
    # Transform the original features to categorical
    for f in obj_feat:
        _df[f] = _df[f].astype("category")
        _df[f + "Int"] = _df[f].cat.rename_categories(range(_df[f].nunique())).astype(int)
        _df.loc[_df[f].isnull(), f + "Int"] = np.nan  # fix NaN conversion
        _df[f] = _df[f + "Int"]
        del _df[f + "Int"]
    return _df


def fill_missing_values(_df, dis_features, cont_features):
    # for discrete features we will use 'most_frequent' strategy
    imp_discrete = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    _df[dis_features] = imp_discrete.fit_transform(_df[dis_features].values)

    # for continuous features we will use 'mean' strategy
    imp_continuous = Imputer(missing_values='NaN', strategy='mean', axis=0)
    _df[cont_features] = imp_continuous.fit_transform(_df[cont_features].values)
    return _df


def drop_missing_values(_df):
    _df.dropna(inplace=True)


def uniform_to_normal(df, continuous_features):
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[continuous_features].dropna()), columns=continuous_features)
    uniform = set()
    alpha = 0.05

    for c in continuous_features:
        statistic, pvalue = kstest(df_scaled[c], scipy.stats.uniform().cdf)
        if statistic < alpha:
            uniform.add(c)

    zero_to_one = [f for f in uniform if
                   df[f].min() > 0 and df[f].min() < 0.001 and df[f].max() < 1 and df[f].max() > 0.999]
    zero_to_ten = [f for f in uniform if
                   df[f].min() > 0 and df[f].min() < 0.01 and df[f].max() < 10 and df[f].max() > 9.99]
    zero_to_hundred = [f for f in uniform if
                       df[f].min() > 0 and df[f].min() < 0.1 and df[f].max() < 100 and df[f].max() > 99.9]
    for f in uniform:
        min = 0 if f in zero_to_one or f in zero_to_ten or f in zero_to_hundred else df[f].min()
        max = 1 if f in zero_to_one else (10 if f in zero_to_ten else 100 if f in zero_to_hundred else df[f].max())
        df[f] = df[f].map(lambda x: norm.ppf((x - min) / (
        max - min)))  # we could use df_scaled but this should give us better results since what we think are the actual min and max, and not the observed min and max

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return uniform


def z_score_scaling(_df, continuous_features):
    scaler = preprocessing.StandardScaler().fit(_df[continuous_features])
    _df[continuous_features] = scaler.transform(_df[continuous_features])


def reduce_Most_Important_Issue(_df):
    _df['Most_Important_Issue'] = _df['Most_Important_Issue'].map(lambda x: 0 if x in [0, 1, 2, 4, 7] else x)


def get_score(X, Y, clf, kf):
    score_sum = 0
    for k, (train_index, test_index) in enumerate(kf):
        clf.fit(X[train_index], Y[train_index])
        acc = clf.score(X[test_index], Y[test_index])
        score_sum += acc
    return score_sum / kf.n_folds

def fill_f1_by_f2_discrete(df, f1, f2):
    df_nonan = df.dropna()
    f2_to_f1_values = {}
    rows_to_complete = df[f1].isnull() & df[f2].notnull()
    for f2_value in df_nonan[f2].unique():
        f2_to_f1_values[f2_value] = df_nonan[f1][df_nonan[f2] == f2_value].unique()[0]
    df[f1][rows_to_complete] = df[f2][rows_to_complete].map(f2_to_f1_values)


def f1_determine_f2(df_nonan, f1, f2):
    f1_values = df_nonan[f1].unique()
    flag = True
    for f1_value in f1_values:
        if len(df_nonan[f2][df_nonan[f1] == f1_value].unique()) > 1:
            flag = False
            break
    return flag


def prepare_data(continuous_features, features_to_keep, df):
    df = mark_negative_values_as_nan(df)
    df = outlier_detection(df, continuous_features)
    df = categorical_features_transformation(df)
    reduce_Most_Important_Issue(df)
    # fill missing values by correlated features.
    fill_f1_by_f2_linear(df, 'Yearly_ExpensesK', 'Avg_monthly_expense_on_pets_or_plants')
    fill_f1_by_f2_linear(df, 'Yearly_IncomeK', 'Avg_size_per_room')
    fill_f1_by_f2_linear(df, 'Yearly_IncomeK',
                         'Political_interest_Total_Score')  # not perfectly corelated, but better then nothing
    fill_f1_by_f2_linear(df, 'Overall_happiness_score',
                         'Political_interest_Total_Score')  # not perfectly corelated, but better then nothing
    fill_f1_by_f2_discrete(df, 'Most_Important_Issue', 'Last_school_grades')
    fill_f1_by_f2_linear(df, 'Avg_Residancy_Altitude', 'Avg_monthly_expense_when_under_age_21')
    fill_f1_by_f2_discrete(df, 'Will_vote_only_large_party', 'Looking_at_poles_results')
    if 'Vote' in df.columns:
        fill_f1_by_f2_discrete(df, 'Financial_agenda_matters', 'Vote')
    else:
        rows_to_complete = df['Financial_agenda_matters'].isnull()
        # Financial_agenda_matters=true for parties 2,5,6,8. Most_Important_Issue(after reduction)>0 for parties 2,5,8. We're missing party 6 voters here, but that's the best we can do
        df['Financial_agenda_matters'][rows_to_complete] = df['Most_Important_Issue'][rows_to_complete].map(
            lambda x: 1 if x > 0 else 0)

    # safety net
    for c in features_to_keep:
        rows_to_fix = df[c].isnull()
        for row, value in enumerate(rows_to_fix):
            if value:
                if 'Vote' in df.columns:
                    df[c][row] = df[df.Vote == df.Vote[row]][c].mean()
                else:
                    df[c][row] = df[c].mean()  # TODO maybe we can do something better here
    z_score_scaling(df, continuous_features)
    # binaries
    df['Will_vote_only_large_party'] = df['Will_vote_only_large_party'].map(lambda x: 1 if x == 1 else -1)
    df['Financial_agenda_matters'] = df['Financial_agenda_matters'].map(lambda x: 1 if x == 1 else -1)
    # nominal
    dummies_df = pd.get_dummies(df['Most_Important_Issue'], prefix='Most_Important_Issue')
    df = pd.concat([df, dummies_df], axis=1, join='inner').drop('Most_Important_Issue', axis=1)
    return df


###############################################################################
########################### MAIN ##############################################
###############################################################################

def load_and_prepare_data():
    features_to_keep = ['Yearly_ExpensesK', 'Yearly_IncomeK', 'Overall_happiness_score', 'Avg_Residancy_Altitude',
                        'Will_vote_only_large_party', 'Financial_agenda_matters', 'Most_Important_Issue']
    continuous_features = ['Yearly_ExpensesK', 'Yearly_IncomeK', 'Overall_happiness_score', 'Avg_Residancy_Altitude']
    labeled = pd.read_csv('dataset/ElectionsData.csv')
    unlabeled = pd.read_csv('dataset/ElectionsData_Pred_Features.csv')
    l_encoder = label_encoder(labeled)
    pickle.dump(l_encoder, open('encoder.pickle', 'w'))
    labeled = prepare_data(continuous_features, features_to_keep, labeled)
    unlabeled = prepare_data(continuous_features, features_to_keep, unlabeled)
    features_to_keep = ['Yearly_ExpensesK', 'Yearly_IncomeK', 'Overall_happiness_score', 'Avg_Residancy_Altitude',
                        'Will_vote_only_large_party', 'Financial_agenda_matters', 'Most_Important_Issue_0.0',
                        'Most_Important_Issue_3.0', 'Most_Important_Issue_5.0', 'Most_Important_Issue_6.0']
    labeled = labeled[features_to_keep + ['Vote']]
    unlabeled = unlabeled[features_to_keep + ['IdentityCard_Num']]
    return labeled, unlabeled, l_encoder



def cross_validation(_df, classifiers):
    n_folds = 10
    kf = KFold(n=len(_df), n_folds=n_folds, shuffle=True)
#     train_sets=list()
#     test_sets=list()
#     for k, (train_index, test_index) in enumerate(kf):
#         # we want to make up for parties with little voters, as the distribution in the test set might be different
#         # it's important to do it after the division to train and test, otherwise we might (almost certainly) have the same line in the train and test.
#         train_sets.append(bootstrap(_df.iloc[train_index]))
#         test_sets.append(bootstrap(_df.iloc[test_index]))
#
#     pickle.dump(train_sets, open('train_sets.pickle', 'w'))
#     pickle.dump(test_sets, open('test_sets.pickle', 'w'))
    train_sets=pickle.load(open('train_sets.pickle'))
    test_sets=pickle.load(open('test_sets.pickle'))

    scores = {}
    for name, clf in classifiers.iteritems():
        score_sum = 0
        for k in xrange(n_folds):
            clf.fit(train_sets[k].drop('Vote', axis=1).values, train_sets[k].Vote.values)
            acc = clf.score(test_sets[k].drop('Vote', axis=1).values, test_sets[k].Vote.values)
            score_sum += acc
        print("{0} average score: {1:.5}".format(name, score_sum / kf.n_folds))
        scores[name] = score_sum / kf.n_folds

    name, clf = 'MyClassifier2', MyClassifier2(scores)
    score_sum = 0

    for k in xrange(n_folds):
        clf.fit(train_sets[k].drop('Vote', axis=1).values, train_sets[k].Vote.values)
        acc = clf.score(test_sets[k].drop('Vote', axis=1).values, test_sets[k].Vote.values)
        score_sum += acc
    print("{0} average score: {1:.5}".format(name, score_sum / kf.n_folds))
    # scores[name] = score_sum / kf.n_folds

    return scores


def cross_validation_wo_bootstrap(_df, classifiers):
    n_folds = 10
    scores = {}

    for name, clf in classifiers.iteritems():
        clf_scores = cross_val_score(clf, _df.drop('Vote', axis=1), _df.Vote.values, cv=n_folds)
        print("{0} average score: {1:.5}".format(name, clf_scores.mean()))
        scores[name] = clf_scores.mean()

    kf = KFold(n=len(_df), n_folds=n_folds, shuffle=True)
    name, clf = 'MyClassifier2', MyClassifier2(scores)
    score_sum = 0
    for k, (train_index, test_index) in enumerate(kf):
        clf.fit(_df.iloc[train_index].drop('Vote', axis=1).values, _df.iloc[train_index].Vote.values)
        acc = clf.score(_df.iloc[test_index].drop('Vote', axis=1).values, _df.iloc[test_index].Vote.values)
        score_sum += acc
    print("{0} average score: {1:.5}".format(name, score_sum / kf.n_folds))

    return scores


def bootstrap(df):
    df = df.reset_index()
    rows_to_select = range(len(df))
    vote_counts = df.Vote.value_counts()
    for party in xrange(10):
        for i in xrange(max(vote_counts) - vote_counts[party]):
            rows_to_select.append(random.choice(df[df.Vote == party].index))

    return df.iloc[rows_to_select].drop('index', axis=1)


class MyClassifier(object):
    def __init__(self, scores):
        self.scores = scores
        self.classifiers = {}

    def predict(self, X):
        proba=self.predict_proba(X)
        proba.fillna(0, inplace=True)
        return [max(row_scores.iteritems(), key=operator.itemgetter(1))[0] for row_num, row_scores in proba.iterrows()]

    def predict_proba(self, X):
        prediction_scores = defaultdict(lambda: defaultdict(float))
        if len(self.values)==1:
            for i in xrange(len(X)):
                prediction_scores[i][self.values[0]]=1
        else:
            for name, classifier in self.classifiers.iteritems():
                self.update_prediction_scores(X, classifier, name, prediction_scores)

        ret_val = pd.DataFrame(prediction_scores)
        ret_val = ret_val/ret_val.sum()
        return ret_val.transpose()

    def update_prediction_scores(self, X, classifier, name, prediction_scores):
        predictions = classifier.predict(X)
        for i in xrange(len(X)):
            prediction_scores[i][predictions[i]] += self.scores[name]

    def score(self, X, y):
        right = 0.
        predictions = self.predict(X)

        if predictions:
            for i in xrange(len(X)):
                if predictions[i] == y[i]:
                    right += 1
        return right / len(y)

    def fit(self, X, y):
        self.values = sorted(set(y))
        if len(self.values)>1:
            for name, classifier_gen in CLASSIFIERS_GENERATORS.iteritems():
                classifier = classifier_gen()
                classifier.fit(X, y)
                self.classifiers[name] = classifier


class MyClassifier2(MyClassifier):
    def update_prediction_scores(self, X, classifier, name, prediction_scores):
        if hasattr(classifier, 'predict_proba') and is_valid_classifier(classifier, X):
            predictions = classifier.predict_proba(X)
            for row_num in xrange(len(X)):
                for i in xrange(len(predictions[row_num])):
                    prediction_scores[row_num][self.values[i]] += predictions[row_num][i]*self.scores[name]


def is_valid_classifier(clf, test):
    return not hasattr(clf, 'n_neighbors') or clf.n_neighbors <= len(test)


def evaluate_classifier(train, test, classifiers, _bootstrap):
    scores = {}

    if _bootstrap:
        train = bootstrap(train)
        test = bootstrap(test)

    for name, clf in classifiers.iteritems():
        if is_valid_classifier(clf, test):
            if len(train.Vote.unique()) == 1 and train.Vote.unique() == test.Vote.unique():
                scores[name] = 1
            else:
                clf.fit(train.drop('Vote', axis=1).values, train.Vote.values)
                score = clf.score(test.drop('Vote', axis=1).values, test.Vote.values)
                scores[name] = score
        else:
            scores[name] = 0

    name, clf = 'MyClassifier2', MyClassifier2(scores)
    clf.fit(train.drop('Vote', axis=1).values, train.Vote.values)
    scores[name] = clf.score(test.drop('Vote', axis=1).values, test.Vote.values)

    return scores


def get_best_cluster(train, test, clusters):
    aics = {}

    for name, cls in clusters.iteritems():
        cls.fit(train.drop('Vote', axis=1).values, train.Vote.values)
        aic = cls.aic(test.drop('Vote', axis=1).values)
        aics[cls] = aic

    best_cluster = min(aics.iteritems(), key=operator.itemgetter(1))[0]
    print 'best cluster: GMM, n_components: ' + str(best_cluster.n_components)
    return best_cluster


def evaluate_clustering(train, test, clusters, classifiers):
    train = bootstrap(train)
    test = bootstrap(test)
    cls = get_best_cluster(train, test, clusters)
    cls.fit(train.drop('Vote', axis=1).values, train.Vote.values)
    train_pred = cls.predict(train.drop('Vote', axis=1).values)
    test_pred = cls.predict(test.drop('Vote', axis=1).values)
    right = 0.
    clusters_score = pd.DataFrame(classifiers.items(), columns=['Classifier', 'Score'])
    clusters_score.Score = 0

    for v in xrange(cls.n_components):
        train_cluster = train.iloc[[x for x, y in enumerate(train_pred) if y==v]]
        test_cluster = test.iloc[[x for x, y in enumerate(test_pred) if y==v]]

        if not test_cluster.empty:
            classifiers_score = evaluate_classifier(train_cluster, test_cluster, classifiers, False)
            classifiers_score = pd.DataFrame(classifiers_score.items(), columns=['Classifier', 'Score'])
            clusters_score.Score += classifiers_score.Score * len(test_cluster)

    clusters_score.Score /= len(test)
    return clusters_score


def clustering_cross_validation(train, clusters, classifiers):
    scores = pd.DataFrame(classifiers.items(), columns=['Classifier', 'Score'])
    scores.Score = 0
    kf = KFold(n=len(train), n_folds=5, shuffle=True)
    for k, (train_index, test_index) in enumerate(kf):
        clusters_score = evaluate_clustering(train.iloc[train_index], train.iloc[test_index], clusters, classifiers)
        best_classifier = clusters_score[clusters_score.Score==clusters_score.Score.max()].iloc[0]
        scores.Score += clusters_score.Score
        print 'Fold #' + str(k) + ', total score: ' + str(best_classifier.Score) + ', classifier: ' + str(best_classifier.Classifier)

    scores.Score /= kf.n_folds
    return scores


def get_voters_division_score(train, test, classifiers, _bootstrap):
    counts = pd.DataFrame()
    ret_val = {}

    if _bootstrap:
        train = bootstrap(train)
        test = bootstrap(test)

    counts['real'] = test.Vote.value_counts()
    for name, classifier in classifiers.iteritems():
        if hasattr(classifier, 'predict_proba'):
            classifier.fit(train.drop('Vote', axis=1), train.Vote.values)
            proba = pd.DataFrame(classifier.predict_proba(test.drop('Vote', axis=1)))
            counts['predicted'] = proba.sum()
            counts['difference'] = counts.real - counts.predicted
            ret_val[name] = counts.difference.abs().sum()/2/len(test)
        else:
            ret_val[name] = 0

    return ret_val


def voters_division_cv(df, classifiers, _bootstrap):
    scores = defaultdict(int)
    n_folds = 5
    kf = KFold(n=len(df), n_folds=n_folds, shuffle=True)

    for k, (train_index, test_index) in enumerate(kf):
        train = df.iloc[train_index]
        test = df.iloc[test_index]

        for name, score in get_voters_division_score(train, test, classifiers, _bootstrap).iteritems():
            scores[name] += score/n_folds

    for name, score in scores.iteritems():
        print name + " score: " + str(score)

    return scores


def main():
    # labeled, unlabeled, l_encoder = load_and_prepare_data()
    # labeled.to_csv('dataset/labeled.csv', index=False)
    # unlabeled.to_csv('dataset/unlabeled.csv', index=False)
    labeled = pd.read_csv('dataset/labeled.csv')
    unlabeled = pd.read_csv('dataset/unlabeled.csv')
    l_encoder = pickle.load(open('encoder.pickle'))

    train, test = train_test_split(labeled, test_size=0.4)
    test, validation = train_test_split(test, test_size=0.5)

    print '\n\nCross validation without bootstrap:'
    cross_validation_wo_bootstrap(train, CLASSIFIERS)

    print '\n\ntrain vs test without bootstrap:'
    scores = evaluate_classifier(train, test, CLASSIFIERS, False)
    best_classifier = max(scores.iteritems(), key=operator.itemgetter(1))
    print 'best classifier: ' + best_classifier[0] + ', score: ' + str(best_classifier[1])
    print scores

    print '\n\nCross validation with bootstrap:'
    scores = cross_validation(train, CLASSIFIERS)

    print '\n\ntrain vs test with bootstrap:'
    scores = evaluate_classifier(train, test, CLASSIFIERS, True)
    best_classifier = max(scores.iteritems(), key=operator.itemgetter(1))
    print 'best classifier: ' + best_classifier[0] + ', score: ' + str(best_classifier[1])
    print scores

    print '\n\nClustering CV'
    clustering_cv_average_score = clustering_cross_validation(train, CLUSTERS, CLASSIFIERS)
    best_classifier = clustering_cv_average_score[clustering_cv_average_score.Score==clustering_cv_average_score.Score.max()].iloc[0]
    print 'best classifier: ' + best_classifier.Classifier + ', score: ' + str(best_classifier.Score)
    print clustering_cv_average_score

    print '\n\nClustering train vs test'
    clustering_score = evaluate_clustering(train, test, CLUSTERS, CLASSIFIERS)
    best_classifier = clustering_score[clustering_score.Score==clustering_score.Score.max()].iloc[0]
    print 'best classifier: ' + str(best_classifier.Classifier) + ', score: ' + str(best_classifier.Score)
    print clustering_score

    # division of voters:
    print '\n\ndivision of voters: cv with bootstrap'
    avg_error = voters_division_cv(train, CLASSIFIERS, True)
    best_classifier = min(avg_error.iteritems(), key=operator.itemgetter(1))
    print 'best classifier: ' + best_classifier[0] + ', avg error: ' + str(best_classifier[1])

    print '\n\ndivision of voters: cv without bootstrap'
    avg_error = voters_division_cv(train, CLASSIFIERS, False)
    best_classifier = min(avg_error.iteritems(), key=operator.itemgetter(1))
    print 'best classifier: ' + best_classifier[0] + ', avg error: ' + str(best_classifier[1])

    print '\n\ndivision of voters: train vs test with bootstrap'
    for name, score in get_voters_division_score(train, test, CLASSIFIERS, True).iteritems():
        print name + " error: " + str(score)

    print '\n\ndivision of voters: train vs test without bootstrap'
    for name, score in get_voters_division_score(train, test, CLASSIFIERS, False).iteritems():
        print name + " error: " + str(score)

    bootstrap_labeled = bootstrap(labeled)

    # predict votes
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15), n_estimators=1000)
    clf.fit(bootstrap_labeled.drop('Vote', axis=1).values, bootstrap_labeled.Vote.values)
    prediction = pd.DataFrame(unlabeled['IdentityCard_Num'])
    predict = clf.predict(unlabeled.drop('IdentityCard_Num', axis=1))
    prediction['PredictVote'] = l_encoder.inverse_transform(predict)
    prediction.to_csv('predictions.csv', index=False)

    division_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15), n_estimators=100)
    division_clf.fit(bootstrap_labeled.drop('Vote', axis=1).values, bootstrap_labeled.Vote.values)
    proba = pd.DataFrame(division_clf.predict_proba(unlabeled.drop('IdentityCard_Num', axis=1)))
    proba.columns = l_encoder.classes_
    counts = pd.DataFrame()
    counts['predicted'] = proba.sum()
    counts['percentage'] = counts['predicted'] / len(unlabeled) * 100
    print counts


if __name__ == "__main__":
    main()
