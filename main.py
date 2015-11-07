# -*- coding: utf-8 -*-
import pandas as pd

def convert_to_bool(df, *column_names):
    for column_name in column_names:
        df[column_name] = df[column_name].map(lambda x: 1 if x == "Yes" else (-1 if x == 'No' else None))

def identify_and_set_correct_types(df):
    # boolean columns: Looking_at_poles_results Married Will_vote_only_large_party Financial_agenda_matters Voting_Time Gender
    convert_to_bool(df, 'Looking_at_poles_results', 'Married', 'Will_vote_only_large_party', 'Financial_agenda_matters')
    df['Gender'] = df['Gender'].map(lambda x: 1 if x == 'Male' else (-1 if x == 'Female' else None))    
    
    # Ordered Numinal columns: Age_group
    df['Age_group'] = df['Age_group'].map(lambda x: 1 if x == "Below_30" else (2 if x == "30-45" else (3 if x == "45_and_up" else None)))
    df['Voting_Time'] = df['Voting_Time'].map(lambda x: 1 if x == 'By_16:00' else (2 if x == 'After_16:00' else None))
    
    # Unordered Numinal columns: Most_Important_Issue Main_transportation  Occupation

def print_missing_values(df):
    for col in df.columns.values:
        print '%s: %s missing values' % (col, df[col].isnull().sum().sum())

def main():
    df = pd.read_csv('dataset/ElectionsData.csv')
    identify_and_set_correct_types(df)

    print df.head()
    print_missing_values(df)
    
if __name__ == "__main__":
    main()
