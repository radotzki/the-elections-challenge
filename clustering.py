from sklearn.cluster import KMeans
# from sklearn.mixture import GMM
import numpy
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


def main():
    df = pd.read_csv('./dataset/transformed_train.csv')
    df['Will_vote_only_large_party'] = df['Will_vote_only_large_party'].map(lambda x: 1 if x==1 else -1)
    df['Financial_agenda_matters'] = df['Financial_agenda_matters'].map(lambda x: 1 if x==1 else -1)

    n_clusters = 100
    cls = KMeans(n_clusters=n_clusters)
    res = cls.fit_predict(df.drop('Vote', axis=1))

    for v in xrange(n_clusters):
        sum_ = sum(get_votes_count_by_party(df.iloc[[x for x, y in enumerate(res) if y==v]]).itervalues())
        print max({k: (v + .0) / sum_ for k, v in get_votes_count_by_party(df.iloc[[x for x,y in enumerate(res) if y==v]]).iteritems()}.itervalues())

    coalition = set()
    minimal_coalitions = []
    get_minimal_coalitions(coalition, 0, minimal_coalitions)


    for coalition in minimal_coalitions:
        evaluate_coalition(df.drop('Most_Important_Issue', axis=1), coalition)


    values = [int(x) for x in df['Most_Important_Issue'].unique()]
    values.sort()
    for i in values:
        df['Most_Important_Issue_'+str(i)] = df['Most_Important_Issue'].map(lambda x: 1 if x==i else 0)
    df.drop('Most_Important_Issue', axis=1, inplace=True)
    df.describe()

if __name__ == "__main__":
    main()

