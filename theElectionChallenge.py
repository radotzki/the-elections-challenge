# -*- coding: utf-8 -*-

###############################################################################
################## Data preparation ###########################################
###############################################################################
import numpy as np
from sklearn.preprocessing import Imputer
from scipy.stats import norm
from scipy.stats import shapiro
from sklearn import preprocessing

def array_diff(a, b):
    b = set(b)
    return [aa for aa in a if aa not in b]
        
def split_features_by_type(_df):    
    all_features = [c for c in _df.columns if c!='Vote']
    discrete_features = [c for c in _df.columns if len(_df[c].unique())<=20 and c!='Vote']
    continuous_features = array_diff(all_features, discrete_features)
    categorical_features = list(_df.keys()[_df.dtypes.map(lambda x: x=='object')])
    return [all_features, discrete_features, continuous_features, categorical_features]
    
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
    for f in positive_features:
        _df[f] = _df[f].map(lambda x: x if x >= 0 else np.nan)
    
def outlier_detection(_df):
    """
    for all continuous features: keep only values that are within +3 to -3 standard deviations, otherwise set nan
    """
    for f in CONTINUOUS_FEATURES:
        std = _df[f].std()
        mean = _df[f].mean()
        _df[f] = _df[f].map(lambda x: x if np.abs(x-mean)<=(3*std) else np.nan)
        
def fill_missing_values_in_linear_depended_features(_df):
    # TODO: make a decision by an automated task (pearson correlation)
    """
    Avg_Residancy_Altitude and Avg_monthly_expense_when_under_age_21 are linear depended in each other.
    We can use that in order to fill the missing values in Avg_monthly_expense_when_under_age_21
    """
    xValues = _df['Avg_Residancy_Altitude'].values
    yValues = _df['Avg_monthly_expense_when_under_age_21'].values
    m = (yValues[0] - yValues[1]) / (xValues[0] - xValues[1])
    const = yValues[0] - (m * xValues[0])

    def get_avg_monthly_expense_when_under_age_21(avg_Residancy_Altitude_value):
        return const + (m * avg_Residancy_Altitude_value)

    def fill_avg_monthly_expense_when_under_age_21(row):
        if row['Avg_Residancy_Altitude'] >=0:
            row['Avg_monthly_expense_when_under_age_21'] = get_avg_monthly_expense_when_under_age_21(row['Avg_Residancy_Altitude'])
        return row

    _df = _df.apply(fill_avg_monthly_expense_when_under_age_21, axis=1)

def drop_redundancy_features(_df, redundancy_features_arr):
    all_features = array_diff(ALL_FEATURES, redundancy_features_arr)
    discrete_features = array_diff(DISCRETE_FEATURES, redundancy_features_arr)
    continuous_features = array_diff(CONTINUOUS_FEATURES, redundancy_features_arr)
    categorial_features = array_diff(CATEGORICAL_FEATURES, redundancy_features_arr)
    _df.drop(redundancy_features_arr, axis=1, inplace=True)
    return [all_features, discrete_features, continuous_features, categorial_features]

def categorical_features_tranformation(_df):

    # Identify which of the original features are objects
    ObjFeat=_df.keys()[_df.dtypes.map(lambda x: x=='object')]

    # Transform the original features to categorical
    for f in ObjFeat:
        _df[f] = _df[f].astype("category")
        _df[f+"Int"] = _df[f].cat.rename_categories(range(_df[f].nunique())).astype(int)
        _df.loc[_df[f].isnull(), f+"Int"] = np.nan #fix NaN conversion
        _df[f]=_df[f+"Int"]
        del _df[f+"Int"]

def fill_missing_values(_df):
    # for discrete features we will use 'most_frequent' strategy
    imp_discrete = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    _df[DISCRETE_FEATURES] = imp_discrete.fit_transform(_df[DISCRETE_FEATURES].values)
    
    # for continuous features we will use 'mean' strategy
    imp_continuous = Imputer(missing_values='NaN', strategy='mean', axis=0)
    _df[CONTINUOUS_FEATURES] = imp_continuous.fit_transform(_df[CONTINUOUS_FEATURES].values)
    
def drop_missing_values(_df):
    _df.dropna(inplace=True)

def uniform_to_normal(_df):
    i=0
    uniform =[]
    for c in ALL_FEATURES:
        if c not in DISCRETE_FEATURES:
            #test for normal distribution
            v=shapiro(_df[c])[1]
            print str(v) + ": " + c
            if v>0:
                uniform.append(c)
        i+=1   
        
    zero_to_one = [f for f in uniform if _df[f].min()>0 and _df[f].min()<0.001 and _df[f].max()<1 and _df[f].max()>0.999]
    zero_to_ten = [f for f in uniform if _df[f].min()>0 and _df[f].min()<0.01 and _df[f].max()<10 and _df[f].max()>9.99]
    zero_to_hundred = [f for f in uniform if _df[f].min()>0 and _df[f].min()<0.1 and _df[f].max()<100 and _df[f].max()>99.9]
    for f in uniform:    
        min= 0 if f in zero_to_one or f in zero_to_ten or f in zero_to_hundred else _df[f].min()
        max= 1 if f in zero_to_one else (10 if f in zero_to_ten else 100 if f in zero_to_hundred else _df[f].max())
        _df[f] = _df[f].map(lambda x: norm.ppf((x-min)/(max-min))) 
    
    _df.replace([np.inf,-np.inf], np.nan, inplace=True)
    _df.dropna(inplace=True)

def z_score_scaling(_df):
    scaler = preprocessing.StandardScaler().fit(_df[CONTINUOUS_FEATURES])
    _df[CONTINUOUS_FEATURES] = scaler.transform(_df[CONTINUOUS_FEATURES])

def reduce_last_school_grades(_df):
    # TODO: make a decision by an automated task
    _df['Last_school_grades'] = _df['Last_school_grades'].map(lambda x: 60 if x >= 60 else x)        
    
###############################################################################
########################### Filters ###########################################
###############################################################################
from sklearn.feature_selection import chi2, f_classif

def chi2_filter(_df):
    alpha = 0.05
    X = _df.drop(['Vote'], axis=1).values
    Y = _df.Vote.values
    v=chi2(X, Y)[1]
    i=0

    for c in ALL_FEATURES:
        if c in DISCRETE_FEATURES:
            if v[i]<alpha:
                FEATURES_TO_KEEP.append(c)            
        i+=1

def anova_filter(_df):
    alpha = 0.05
    non_categorical = array_diff(ALL_FEATURES, CATEGORICAL_FEATURES)
    X = _df[non_categorical].values
    Y = _df.Vote.values
    v=f_classif(X, Y)[1]
    i=0
    
    for c in _df[non_categorical].columns:
        if v[i]<alpha:
            FEATURES_TO_KEEP.append(c)            
        i+=1
        
###############################################################################
########################### Wrappers ##########################################
###############################################################################
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC

def wrappersTest(X, Y, kf): 
    classifiers = {
        "Nearest Neighbors": KNeighborsClassifier(15),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Perceptron": Perceptron(n_iter=50),
    #     "Linear SVM OVO": SVC(kernel="linear", C=1),
        "Linear SVM OVR": LinearSVC(C=1),
        "Random Forest": RandomForestClassifier(n_estimators = 3)
    }
    res = {}
    for name, clf in classifiers.iteritems():
        score_sum=0 
        print 'start ' + str(name) + ' test..'
        for k, (train_index, test_index) in enumerate(kf):            
            clf.fit(X[train_index], Y[train_index])            
            acc = clf.score(X[test_index],Y[test_index])
            score_sum += acc                 
        print("{0} average score: {1:.5}".format(name, score_sum/kf.n_folds))
        res[name] = score_sum/kf.n_folds
    return res

def evaulate_features(_df, similar_features):    
    n_folds=5
    kf = KFold(n=len(_df), n_folds=n_folds)
    Y = _df.Vote.values
    
    res = {}
    print 'Wrappers score with all selected features:'
    res['all'] = wrappersTest(_df[FEATURES_TO_KEEP].values, Y, kf)
    
    print 'Wrappers score without similar_features:'
    res['withou similar_features'] = wrappersTest(_df[FEATURES_TO_KEEP].drop(similar_features, axis=1).values, Y, kf)

    for s in similar_features:   
        print 'Wrappers score without ' + str(s) + ':'
        res[s] = wrappersTest(_df[FEATURES_TO_KEEP].drop(s, axis=1).values, Y, kf)
        
    print pd.DataFrame.from_dict(res)    
    return res   
    
def SFS(df, label, classifier, max_out_size, n_folds=5):
    kf = KFold(n=len(df), n_folds=n_folds)
    labels = df[label].values
    selected_features = []
    not_selected_features = list(df.columns)
    not_selected_features.remove(label)
    last_score = 0
    while len(selected_features) < max_out_size and len(not_selected_features)>0:
        max = 0
        for feature in not_selected_features:
            score = get_score(df[selected_features+[feature]].values, labels, classifier, kf)
            if score > max:
                max=score
                best_feature=feature
        if max<last_score:
            print 'no improvemant by adding any feature'
            break
        selected_features.append(best_feature)
        not_selected_features.remove(best_feature)
        last_score=max
        print 'selected feature: ' + best_feature + ' with score ' + str(max)
    return selected_features
        
def get_score(X, Y, clf, kf):
    score_sum=0
    for k, (train_index, test_index) in enumerate(kf):
        clf.fit(X[train_index], Y[train_index])
        acc = clf.score(X[test_index],Y[test_index])
        score_sum += acc
    return score_sum/kf.n_folds
    
###############################################################################
########################### MAIN ##############################################
###############################################################################
        
import pandas as pd

df = pd.read_csv('dataset/ElectionsData.csv')

ALL_FEATURES, DISCRETE_FEATURES, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES = split_features_by_type(df)
FEATURES_TO_KEEP=[]

mark_negative_values_as_nan(df)
outlier_detection(df)
fill_missing_values_in_linear_depended_features(df)
ALL_FEATURES, DISCRETE_FEATURES, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES = drop_redundancy_features(df, ['Avg_Residancy_Altitude'])
mark_negative_values_as_nan(df)
categorical_features_tranformation(df)
fill_missing_values(df)
reduce_last_school_grades(df)
chi2_filter(df)
uniform_to_normal(df)
z_score_scaling(df)
anova_filter(df)

# TODO: make a decision by an automated task
similar_features = ['Avg_monthly_expense_when_under_age_21', 'AVG_lottary_expanses', 'Garden_sqr_meter_per_person_in_residancy_area']
evaulate_features(df, similar_features)

sfs=SFS(df, 'Vote', RandomForestClassifier(n_estimators = 3), 18)
print "features in sfs we didn't select:"
for f in sfs:
    if f not in FEATURES_TO_KEEP:
        print f
print ''
print "features we selected and sfs didn't:"
for f in FEATURES_TO_KEEP:
    if f not in sfs:
        print f
        
# TODO: remove features according to the wrappers result

# TODO: suddenly we get that 'Occupation_Satisfaction' is part of the selected features
print FEATURES_TO_KEEP