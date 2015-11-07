# -*- coding: utf-8 -*-
import pandas as pd

def convert_to_bool(df, *column_names):
    for column_name in column_names:
        df[column_name] = df[column_name].map(lambda x: 1 if x == "Yes" else (-1 if x == 'No' else None))

def identify_and_set_correct_types(df):
    # boolean columns: Looking_at_poles_results Married Will_vote_only_large_party Financial_agenda_matters Voting_Time Gender
    convert_to_bool(df, 'Looking_at_poles_results', 'Married', 'Will_vote_only_large_party', 'Financial_agenda_matters')
    df['Gender'] = df['Gender'].map(lambda x: 1 if x == 'Male' else (-1 if x == 'Female' else None))    
    
    # Ordered nominal columns: Age_group
    df['Age_group'] = df['Age_group'].map(lambda x: 1 if x == "Below_30" else (2 if x == "30-45" else (3 if x == "45_and_up" else None)))
    df['Age_group'] = df['Age_group'].astype('category', ordered=True)
    df['Voting_Time'] = df['Voting_Time'].map(lambda x: 1 if x == 'By_16:00' else (2 if x == 'After_16:00' else None))
    df['Voting_Time'] = df['Voting_Time'].astype('category', ordered=True)
    
    # Unordered nominal columns: Most_Important_Issue Main_transportation  Occupation
    # we are losing the missing data here
    for col in ['Most_Important_Issue', 'Main_transportation', 'Occupation']:
        partial = pd.get_dummies(df[col], col, '_')
        df = df.join(partial)
        del df[col]
        
    return df
        
def print_missing_values(df):
    print '\n\nMissing values:'
    for col in df.columns.values:
        misCount = df[col].isnull().sum()
        if misCount > 0:
            print '%s: %s missing values' % (col, misCount)
        
def data_imputation(df):
    # method='nearset'
    #return df.interpolate(method='linear')

#   # median
#    for col in df.columns.values:
#        df[col] = df[col].fillna(df[col].median())
        
    return df

def main():
    df = pd.read_csv('dataset/ElectionsData.csv')
    df = identify_and_set_correct_types(df)
    print df.head()
    
#    df = data_imputation(df)
    print_missing_values(df)
    
if __name__ == "__main__":
    main()
