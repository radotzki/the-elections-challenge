# -*- coding: utf-8 -*-
import pandas as pd

def convert_to_bool(df, *column_names):
    for column_name in column_names:
        df[column_name] = df[column_name].map(lambda x: True if x == "Yes" else (False if x == 'No' else None))

def identify_and_set_correct_types(df):
    # boolean columns: Looking_at_poles_results Married Will_vote_only_large_party Financial_agenda_matters
    convert_to_bool(df, 'Looking_at_poles_results', 'Married', 'Will_vote_only_large_party', 'Financial_agenda_matters')

    # Numinal columns: Occupation Main_transportation Most_Important_Issue Age_group Early_voter
    df['Early_voter'] = df['Voting_Time'].map(lambda x: True if x == 'By_16:00' else (False if x == 'After_16:00' else None))
    del df['Voting_Time']
    df['Age_group'] = df['Age_group'].map(lambda x: 1 if x == "Below_30" else (2 if x == "30-45" else 3))

def main():
    df = pd.read_csv('dataset/ElectionsData.csv')
    identify_and_set_correct_types(df)
    
if __name__ == "__main__":
    main()
