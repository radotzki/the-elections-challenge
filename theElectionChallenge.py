# -*- coding: utf-8 -*-

###############################################################################
################## Data preparation ###########################################
###############################################################################
from sets import Set
import numpy as np
from numpy.linalg import lstsq
from sklearn.preprocessing import Imputer
from scipy.stats import norm
from scipy.stats import shapiro
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
import pandas as pd
import scipy
import scipy.interpolate


def array_diff(a, b):
    b = set(b)
    return [aa for aa in a if aa not in b]


def split_features_by_type(_df):
    _df = _df.dropna()
    all_features = [c for c in _df.columns if c != 'Vote']
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
    X = tuple((tuple((pow(x,p) for p in exponents)) for x in Xs))
    y = tuple(((y) for y in Ys))
    x, resids, rank, s = lstsq(X,y)
    return x


def fill_f1_by_f2(df, f1, f2):
    rows_to_complete = df[f1].isnull() & df[f2].notnull()

    df_dropna = df[[f1, f2]].dropna()
    coefs = find_coefficients(df_dropna[f2],df_dropna[f1],range(2)) #linear approximation
    # for i, row in df.iterrows():
    #     if rows_to_complete[i]:
    #         x=df[f1][i]
    #         print x
    #         print y_interp(x)
    #         df[f2][i] = y_interp(df[f1][i])
    df[f1][rows_to_complete] = df[f2][rows_to_complete].map(lambda x: coefs[0] + coefs[1]*x)


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


def uniform_to_normal(_df, continuous_features):
    i = 0
    uniform = []
    for c in continuous_features:
        # test for normal distribution
        v = shapiro(_df[c])[1]
        # v=stats.kstest(df[c].values, 'uniform')
        # TODO: scale first (0-1)
        if v > 0:
            uniform.append(c)
        i += 1

    zero_to_one = [f for f in uniform if
                   _df[f].min() > 0 and _df[f].min() < 0.001 and _df[f].max() < 1 and _df[f].max() > 0.999]
    zero_to_ten = [f for f in uniform if
                   _df[f].min() > 0 and _df[f].min() < 0.01 and _df[f].max() < 10 and _df[f].max() > 9.99]
    zero_to_hundred = [f for f in uniform if
                       _df[f].min() > 0 and _df[f].min() < 0.1 and _df[f].max() < 100 and _df[f].max() > 99.9]
    for f in uniform:
        min = 0 if f in zero_to_one or f in zero_to_ten or f in zero_to_hundred else _df[f].min()
        max = 1 if f in zero_to_one else (10 if f in zero_to_ten else 100 if f in zero_to_hundred else _df[f].max())
        _df[f] = _df[f].map(lambda x: norm.ppf((x - min) / (max - min)))

    _df.replace([np.inf, -np.inf], np.nan, inplace=True)
    _df.dropna(inplace=True)


def z_score_scaling(_df, continuous_features):
    scaler = preprocessing.StandardScaler().fit(_df[continuous_features])
    _df[continuous_features] = scaler.transform(_df[continuous_features])


def reduce_last_school_grades(_df):
    # TODO: make a decision by an automated task
    _df['Last_school_grades'] = _df['Last_school_grades'].map(lambda x: 60 if x >= 60 else x)


###############################################################################
########################### Filters ###########################################
###############################################################################
from sklearn.feature_selection import chi2, f_classif

alpha = 0.05


def chi2_filter(_df, features_to_test):
    ret_val = []
    X = _df.drop(['Vote'], axis=1).values
    Y = _df.Vote.values
    v = chi2(X, Y)[1]
    i = 0

    for c in features_to_test:
        if v[i] < alpha:
            ret_val.append(c)
        i += 1

    return ret_val


def anova_filter(_df, features):
    ret_val=Set()
    X = _df[features].values
    Y = _df.Vote.values
    v = f_classif(X, Y)[1]
    i = 0

    for c in features:
        if v[i] < alpha:
            ret_val.add(c)
        i += 1
    return ret_val


###############################################################################
########################### Wrappers ##########################################
###############################################################################


def wrappersTest(X, Y, kf):
    classifiers = {
        "Nearest Neighbors": KNeighborsClassifier(15),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Perceptron": Perceptron(n_iter=50),
        #     "Linear SVM OVO": SVC(kernel="linear", C=1),
        "Linear SVM OVR": LinearSVC(C=1),
        "Random Forest": RandomForestClassifier(n_estimators=3)
    }
    res = {}
    for name, clf in classifiers.iteritems():
        score_sum = 0
        print 'start ' + str(name) + ' test..'
        for k, (train_index, test_index) in enumerate(kf):
            clf.fit(X[train_index], Y[train_index])
            acc = clf.score(X[test_index], Y[test_index])
            score_sum += acc
        print("{0} average score: {1:.5}".format(name, score_sum / kf.n_folds))
        res[name] = score_sum / kf.n_folds
    return res


def evaulate_features(_df, Y, similar_features):
    n_folds = 5
    kf = KFold(n=len(_df), n_folds=n_folds)

    res = {}
    print 'Wrappers score with all selected features:'
    res['all'] = wrappersTest(_df.values, Y, kf)

    # print 'Wrappers score without similar_features:'
    # res['withou similar_features'] = wrappersTest(_df[FEATURES_TO_KEEP].drop(similar_features, axis=1).values, Y, kf)
    #
    # for s in similar_features:
    #     print 'Wrappers score without ' + str(s) + ':'
    #     res[s] = wrappersTest(_df[FEATURES_TO_KEEP].drop(s, axis=1).values, Y, kf)
    #
    # print pd.DataFrame.from_dict(res)
    return res


def SFS(df, label, classifier, max_out_size, n_folds=5):
    kf = KFold(n=len(df), n_folds=n_folds)
    labels = df[label].values
    selected_features = []
    not_selected_features = list(df.columns)
    not_selected_features.remove(label)
    last_score = 0
    while len(selected_features) < max_out_size and len(not_selected_features) > 0:
        max = 0
        for feature in not_selected_features:
            score = get_score(df[selected_features + [feature]].values, labels, classifier, kf)
            if score > max:
                max = score
                best_feature = feature
        if max < last_score:
            print 'no improvemant by adding any feature'
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


def find_most_correlated(df, features):
    max_cor = 0
    for i in xrange(0, len(features)):
        for j in xrange(i + 1, len(features)):
            pearsonr = scipy.stats.pearsonr(df[features[i]], df[features[j]])
            if pearsonr[1] < alpha:
                cor = abs(pearsonr[0])
                if (cor > max_cor):
                    max_cor = cor
                    max_i = i
                    max_j = j
    if max_cor > 0:
        return [features[max_i], features[max_j], max_cor]
    else:
        return None


###############################################################################
########################### MAIN ##############################################
###############################################################################

def main():
    df = pd.read_csv('dataset/ElectionsData.csv')

    all_features, discrete_features, continuous_features, categorical_features, numeric_features = split_features_by_type(df)
    features_to_keep = Set()
    df = mark_negative_values_as_nan(df)
    df = outlier_detection(df, continuous_features)
    df = categorical_features_transformation(df)
    most_correlated = find_most_correlated(df.dropna(), numeric_features)
    while most_correlated is not None:
        feature1, feature2, cof = most_correlated
        if cof < 0.95:
            break
        # feature1, feature2 = feature2, feature1
        print feature1 + " and " + feature2 + " are correlated by " + str(cof) + ". Filling missing values and dropping " + feature2
        fill_f1_by_f2(df, feature2, feature1)
        df.drop(feature2, axis=1, inplace=True)
        for x in [all_features, discrete_features, continuous_features, categorical_features, numeric_features]:
            if feature2 in x:
                x.remove(feature2)
        most_correlated = find_most_correlated(df.dropna(), numeric_features)

    df = mark_negative_values_as_nan(df)
    df = fill_missing_values(df, discrete_features, continuous_features)
    reduce_last_school_grades(df)
    features_to_keep = features_to_keep.union(chi2_filter(df, categorical_features))
    uniform_to_normal(df, continuous_features)
    # todo test for normality and maybe use log-values
    z_score_scaling(df, continuous_features)
    print len(features_to_keep)
    print features_to_keep
    features_to_keep = features_to_keep.union(anova_filter(df, numeric_features))
    print len(features_to_keep)
    print features_to_keep

    # evaulate_features(df[list(features_to_keep)], df.Vote.values, [])

    sfs = SFS(df, 'Vote', RandomForestClassifier(n_estimators=3), 18)
    print "features in sfs we didn't select:"
    for f in sfs:
        if f not in features_to_keep:
            print f
    print ''
    print "features we selected and sfs didn't:"
    for f in features_to_keep:
        if f not in sfs:
            print f

    # TODO: remove features according to the wrappers result
    print features_to_keep


if __name__ == "__main__":
    main()
