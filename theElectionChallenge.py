# -*- coding: utf-8 -*-

###############################################################################
################## Data preparation ###########################################
###############################################################################
from collections import defaultdict
import numpy as np
from numpy.linalg import lstsq
import pickle
import operator
from sklearn.preprocessing import Imputer, MinMaxScaler
from scipy.stats import norm, kstest
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
import pandas as pd
import scipy
import scipy.interpolate

classifiers = {
    "Decision Tree 10": DecisionTreeClassifier(max_depth=10),
    "Perceptron  100": Perceptron(n_iter=100),
    "Linear SVM OVO": SVC(kernel="linear", C=1),
    # "Linear SVM OVR": LinearSVC(C=1),
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


###############################################################################
########################### Filters ###########################################
###############################################################################
from sklearn.feature_selection import chi2, f_classif

alpha = 0.05


def chi2_filter(_df, features_to_test):
    ret_val = []
    X = _df[features_to_test].values
    Y = _df.Vote.values
    v = chi2(X, Y)[1]
    i = 0

    for c in features_to_test:
        if v[i] < alpha:
            print c + " selected by chi2 with p-value: " + str(v[i])
            ret_val.append(c)
        i += 1

    return ret_val


def anova_filter(_df, features):
    ret_val = set()
    X = _df[features].values
    Y = _df.Vote.values
    v = f_classif(X, Y)[1]
    i = 0

    for c in features:
        if v[i] < alpha:
            print c + " selected by anova with p-value: " + str(v[i])
            ret_val.add(c)
        i += 1
    return ret_val


###############################################################################
########################### Wrappers ##########################################
###############################################################################


def wrappersTest(X, Y, kf):
    res = {}
    for name, clf in classifiers.iteritems():
        score_sum = 0
        # print 'start ' + str(name) + ' test..'
        for k, (train_index, test_index) in enumerate(kf):
            clf.fit(X[train_index], Y[train_index])
            acc = clf.score(X[test_index], Y[test_index])
            score_sum += acc
        print("{0} average score: {1:.5}".format(name, score_sum / kf.n_folds))
        res[name] = score_sum / kf.n_folds
    return res


def evaluate_features(_df, Y):
    n_folds = 5
    kf = KFold(n=len(_df), n_folds=n_folds, shuffle=True)

    res = {}
    # Wrappers score with all selected features:
    res['all'] = wrappersTest(_df.values, Y, kf)
    print pd.DataFrame.from_dict(res)
    return res


def SFS(df, label, classifier, max_out_size=15, n_folds=5, min_improve=0.001):
    kf = KFold(n=len(df), n_folds=n_folds, shuffle=True)
    labels = df[label].values
    selected_features = []
    not_selected_features = list(df.columns)
    not_selected_features.remove(label)
    not_selected_features.remove("split")
    last_score = 0
    while len(selected_features) < max_out_size and len(not_selected_features) > 0:
        max = 0
        for feature in not_selected_features:
            score = get_score(df[selected_features + [feature]].values, labels, classifier, kf)
            if score > max:
                max = score
                best_feature = feature
        if max - last_score < min_improve:
            print 'not enough improvemant by adding any feature'
            break
        selected_features.append(best_feature)
        not_selected_features.remove(best_feature)
        last_score = max
        print 'selected feature: ' + best_feature + ' with score ' + str(max)
    return selected_features


def get_score(X, Y, clf, kf):
    score_sum = 0
    for k, (train_index, test_index) in enumerate(kf):
        clf.fit(X[train_index], Y[train_index])
        acc = clf.score(X[test_index], Y[test_index])
        score_sum += acc
    return score_sum / kf.n_folds


def find_most_correlated(df, features, feature_to_compare):
    votes = df.Vote.unique()
    votes.sort()
    max_cor = 0
    for j in xrange(0, len(features)):
        if (features[j] == feature_to_compare):
            continue
        pearsonr = scipy.stats.pearsonr(df[feature_to_compare], df[features[j]])
        if pearsonr[1] < alpha:
            cor = abs(pearsonr[0])
            if cor == 1:
                return [feature_to_compare, features[j], 1]
            # let's check the correlation in each label separatly, and use the minimum
            # min_per_label_cor = 1
            # for vote in votes:
            #     per_label_pearson = scipy.stats.pearsonr(df[df.Vote == vote][feature_to_compare],
            #                                           df[df.Vote == vote][features[j]])
            #     if (per_label_pearson[1]>= alpha):
            #         min_per_label_cor = 0
            #         break # we don't want to use this
            #     if (abs(per_label_pearson[0])<min_per_label_cor):
            #         min_per_label_cor = abs(per_label_pearson[0])
            # cor=min(cor, min_per_label_cor)
            if (cor > max_cor):
                max_cor = cor
                max_j = j

    if max_cor > 0:
        return [features[max_j], max_cor]
    else:
        return None


def drop_redundant_numeric_features(df, all_features, categorical_features, continuous_features, discrete_features,
                                    numeric_features):
    most_correlated = find_most_correlated(df.dropna(), numeric_features)
    while most_correlated is not None:
        feature1, feature2, cof = most_correlated
        if cof < 0.95:
            break
        # feature1, feature2 = feature2, feature1
        print feature1 + " and " + feature2 + " are correlated by " + str(
            cof) + ". Filling missing values and dropping " + feature2
        fill_f1_by_f2_linear(df, feature2, feature1)
        df.drop(feature2, axis=1, inplace=True)
        for x in [all_features, discrete_features, continuous_features, categorical_features, numeric_features]:
            if feature2 in x:
                x.remove(feature2)
        most_correlated = find_most_correlated(df.dropna(), numeric_features)


def drop_redundant_discrete_features(df, all_features, categorical_features, continuous_features, discrete_features,
                                     numeric_features):
    found = True
    while found:
        found = detect_single_redundant_discrete_feature(all_features, categorical_features, continuous_features, df,
                                                         discrete_features, numeric_features)


def detect_single_redundant_discrete_feature(all_features, categorical_features, continuous_features, df,
                                             discrete_features, numeric_features):
    df_nonan = df.dropna()
    for i in xrange(0, len(discrete_features)):
        f1 = discrete_features[i]
        for j in xrange(i + 1, len(discrete_features)):
            f2 = discrete_features[j]
            if f1_determine_f2(df_nonan, f1, f2) and f1_determine_f2(df_nonan, f2, f1):
                # going to drop f2
                if f2 in numeric_features and f1 not in numeric_features:
                    f1, f2 = f2, f1  # better keep a numeric value
                fill_f1_by_f2_discrete(df, f1, f2)
                print 'Dropping ' + f2 + " because it's equivalent to " + f1
                df.drop(f2, axis=1, inplace=True)
                for x in [all_features, discrete_features, continuous_features, categorical_features, numeric_features]:
                    if f2 in x:
                        x.remove(f2)
                return True
    return False


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


###############################################################################
########################### MAIN ##############################################
###############################################################################

class MyClassifier(object):
    def __init__(self, scored_classifiers):
        self.scored_classifiers = scored_classifiers

    def predict(self, vector):
        prediction_scores = defaultdict(float)
        for classifier, score in self.scored_classifiers.iteritems():
            prediction_scores[classifier.predict(vector)[0]] += score
        return max(prediction_scores.iteritems(), key=operator.itemgetter(1))

    def score(self, X, y):
        right = 0.
        for i, row in X.iterrows():
            if self.predict(row)[0] == y[i]:
                right += 1
        return right / len(y)

    def fit(self, X, y):
        pass

def main():
    train = pd.read_csv('dataset/transformed_train.csv')
    test = pd.read_csv('dataset/transformed_test.csv')
    l_encoder = pickle.load(open('encoder.pickle'))

    features = list(train.columns)
    features.remove('Vote')
    # evaluate_features(train[features], train.Vote.values)
    scored_classifiers = dict()
    for name, classifier in classifiers.iteritems():
        evaluate_classifier_against_test(classifier, features, l_encoder, name, scored_classifiers, test, train)

    classifier = MyClassifier(scored_classifiers.copy())
    evaluate_classifier_against_test(classifier, features, l_encoder, 'Custom classifier', scored_classifiers, test, train)

    # df.Vote = label_decoder(df, l_encoder)
    # features_to_keep.remove("Vote")
    # features_to_keep.remove("split")


def evaluate_classifier_against_test(classifier, features, l_encoder, name, scored_classifiers, test, train):
    classifier.fit(train[list(features)], train.Vote.values)
    score = classifier.score(test[list(features)], test.Vote.values)
    print name + " score: " + str(score)
    scored_classifiers[classifier] = score
    confusion = defaultdict(lambda: defaultdict(int))
    for i, row in test.iterrows():
        prediction = l_encoder.inverse_transform(classifier.predict(row.drop('Vote'))[0])
        actual = l_encoder.inverse_transform(int(row['Vote']))
        confusion[prediction][actual] += 1
    print 'Confusion table:'
    print '      |predicted'
    print '----------------'
    print 'actual|'
    print ''
    print pd.DataFrame.from_dict(confusion).fillna(0)
    print 'min sensitivity: ' + str(min(
        float(actual[prediction]) / sum(x[prediction] for x in confusion.itervalues()) for prediction, actual in
        confusion.iteritems()))
    print 'min percision: ' + str(
        min(float(actual[prediction]) / sum(actual.values()) for prediction, actual in confusion.iteritems()))
    print ''
    print '--------------------------------------------------------------------------------------------------'
    print ''


if __name__ == "__main__":
    main()
