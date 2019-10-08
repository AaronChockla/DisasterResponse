"""
Data Preprocessing for Disaster Response Pipeline Project

Arguments:
    1) CSV file containing messages (e.g., disaster_messages.csv)
    2) CSV file containing categories (e.g., disaster_categories.csv)
    3) SQLite destination database (e.g., DisasterResponse.db)
    
Example:
    python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
"""

import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Function to load message and category data 


    Arguments:
        1) messages_filepath: path to CSV file containing messages
        2) categories_filepath: path to CSV file containing categories

    Output:
        1) df: dataframe of loaded data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories,how='outer', on = 'id')
    
    return df



def clean_data(df):
    """
    Function to clean data
    
    Arguments:
        1) df: dataframe of merged message / category data
    
    Output:
        1) df: cleaned data frame
    """
    categories = df.categories.str.split(';',expand=True)
    
    # extract a list of new column names for categories
    labels = categories.iloc[0].apply(lambda x: x.split('-')[0])
    
    # rename the columns of `categories`
    categories.columns = labels
    
    # Convert category values to dummy values, i.e., 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
        
    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # Remove duplicates
    
    # Define new function to check number of duplicates
    def check_duplicates(df):
        dupes = df.pivot_table(index=['id'], aggfunc='size')
        duplicates = dupes[dupes>1]
        return duplicates.shape[0]
    
    # check duplicates
    check_duplicates(df)

    # drop duplicates
    df.drop_duplicates(subset='id', inplace=True)
        
    # re-check number of duplicates
    check_duplicates(df)
    
    return df




def save_data(df, database_filename):
    """
    Function to save the clean dataset into an sqlite database
    
    Arguments:
        1) df: cleaned data frame
        2) database_filename
    
    Output: None
    """
    engine = create_engine('sqlite:///'+database_filename)
    
    db_file_name = database_filename.split("/")[-1] # extract file name from \
                                                     # the file path
    table_name = db_file_name.split(".")[0]
    
    df.to_sql(table_name, engine, index=False)


def main():
    """
    Master function for data processing
    
    This function implements the ETL pipeline by:
        1) Extracting data from CSV files
        2) Cleaning and pre-processing the data for the ML pipeline
        3) Loading the data into a SQLite database
    """
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()