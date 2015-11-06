# -*- coding: utf-8 -*-
import pandas as pd


def convert_to_bool(df, *column_names):
    for column_name in column_names:
        df[column_name] = df[column_name].map(lambda x: True if x == "Yes" else False)


def main():
    df = pd.read_csv('dataset/ElectionsData.csv')
    convert_to_bool(df, 'Looking_at_poles_results', 'Married', 'Will_vote_only_large_party', 'Financial_agenda_matters')
    df['Early_voter'] = df['Voting_Time'].map(lambda x: True if x == "By_16:00" else False)
    del df['Voting_Time']
    df['Age_group'] = df['Age_group'].map(lambda x: 1 if x == "Below_30" else (2 if x == "30-45" else 3))

if __name__ == "__main__":
    main()
