import math
import pandas as pd
df = pd.read_csv('./dataset/transformed_train.csv')
# df.describe()
def is_majority(coalition):
    count = 0
    for party in coalition:
        count += len(df[df.Vote==party])
    return count > len(df)/2

def evaluate_coalition(coalition):
    print coalition

def is_minimal_coalition(coalition):
    if not is_majority(coalition):
        return False
    for party in coalition:
        if (is_majority(coalition.difference([party]))):
            return False
    return True

def get_coalitions(coalition, i):
    if is_minimal_coalition(coalition):
        evaluate_coalition(coalition)
    else:
        if i==10:
            return
        get_coalitions(coalition.union([i]), i+1)
        get_coalitions(coalition, i+1)

coalition = set()
get_coalitions(coalition, 0)

