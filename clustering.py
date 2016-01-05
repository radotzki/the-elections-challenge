from sklearn.cluster import KMeans
from sklearn.mixture import GMM
import numpy
import numpy as np
import pandas as pd


def is_majority(coalition):
    return len(df[df.Vote.isin(coalition)]) > len(df)/2


def is_minimal_coalition(coalition):
    if not is_majority(coalition):
        return False
    for party in coalition:
        if is_majority(coalition.difference([party])):
            return False
    return True


def get_minimal_coalitions(coalition, i, minimal_coalitions):
    if is_minimal_coalition(coalition):
        minimal_coalitions.append(coalition)
    else:
        if i==10:
            return
        get_minimal_coalitions(coalition.union([i]), i+1, minimal_coalitions)
        get_minimal_coalitions(coalition, i+1, minimal_coalitions)


def get_votes_count_by_party(df):
    parties = {}
    for party in df.Vote.unique():
        parties[party] = len(df[df.Vote == party])
    return parties


def evaluate_coalition(df, coalition):
    coalition_df = df[df.Vote.isin(coalition)].drop('Vote', axis=1)
    coalition_mean = coalition_df.mean()
    opposition_mean = df[-df.Vote.isin(coalition)].drop('Vote',axis=1).mean()
    coalition_df['distance_from_mean'] = numpy.linalg.norm(coalition_df-coalition_mean, axis=1)
    distance_within_coalition = coalition_df['distance_from_mean'].mean()
    distance_between = numpy.linalg.norm(coalition_mean-opposition_mean)
    score = distance_between / distance_within_coalition
    print 'Coalition: ' + str(coalition) + ", size: " + str(len(coalition_df)) + ", score: " + str(score)


def binary_transformation(df):
    df['Will_vote_only_large_party'] = df['Will_vote_only_large_party'].map(lambda x: 1 if x==1 else -1)
    df['Financial_agenda_matters'] = df['Financial_agenda_matters'].map(lambda x: 1 if x==1 else -1)


def dummis_transformation(df):
    dummies_df = pd.get_dummies(df['Most_Important_Issue'], prefix='Most_Important_Issue')
    df = pd.concat([df, dummies_df], axis=1, join='inner')
    df.drop('Most_Important_Issue', axis=1, inplace=True)


def main():
    df = pd.read_csv('./dataset/transformed_train.csv')
    binary_transformation(df)

    # n_clusters = 100
    # cls = KMeans(n_clusters=n_clusters)
    # res = cls.fit_predict(df.drop('Vote', axis=1))
    #
    # for v in xrange(n_clusters):
    #     sum_ = sum(get_votes_count_by_party(df.iloc[[x for x, y in enumerate(res) if y==v]]).itervalues())
    #     print max({k: (v + .0) / sum_ for k, v in get_votes_count_by_party(df.iloc[[x for x,y in enumerate(res) if y==v]]).iteritems()}.itervalues())
    #
    # coalition = set()
    # minimal_coalitions = []
    # get_minimal_coalitions(coalition, 0, minimal_coalitions)
    #
    #
    # for coalition in minimal_coalitions:
    #     evaluate_coalition(df.drop('Most_Important_Issue', axis=1), coalition)

    dummis_transformation(df)

    X_train = df.drop('Vote', axis=1)
    y_train = df.Vote.values

    lowest_aic = np.infty
    aic = []
    # TODO: keep only the commented out lines
    # n_components_range = range(1, 20)
    # cv_types = ['spherical', 'tied', 'diag', 'full']
    n_components_range = [18]
    cv_types = ['diag']

    for cv_type in cv_types:
        for n_components in n_components_range:
            gmm = GMM(n_components=n_components, covariance_type=cv_type)

            # Since we have class labels for the training data, we can
            # initialize the GMM parameters in a supervised manner.
            gmm.means_ = np.array([X_train[y_train == i].mean(axis=0) for i in xrange(n_components)])

            # Fit a mixture of Gaussians with EM
            gmm.fit(X_train)

            aic.append(gmm.aic(X_train))
            print 'n_components: ' + str(n_components) +\
                    ' cv_type: ' + str(cv_type) + \
                    ' aic: ' + str(gmm.aic(X_train)) + \
                    ' bic: ' + str(gmm.bic(X_train))
            if aic[-1] < lowest_aic:
                lowest_aic = aic[-1]
                best_gmm = gmm

    print 'best model:'
    print '\tn_components = ' + str(best_gmm.n_components)
    print '\tcovariance_type = ' + str(best_gmm.covariance_type)
    print '\taic = ' + str(lowest_aic)

    y_train_pred = best_gmm.predict(X_train)
    for v in xrange(best_gmm.n_components):
        cluster = df.iloc[[x for x, y in enumerate(y_train_pred) if y==v]]
        print get_votes_count_by_party(cluster)



if __name__ == "__main__":
    main()

