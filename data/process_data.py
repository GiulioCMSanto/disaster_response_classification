import sys
import pandas as pd
from sqlalchemy import create_engine

def transform_categories(categories_df):
    """
    Makes transformations in the categories data.
    Arguments:
        categories_df: the categories dataframe
    
    Output:
        categories_df_trans: the transformed categories dataframe
    """
    categories_df_trans = categories_df.copy()
    
    #Store rows ids
    ids = categories_df_trans['id']
    
    categories_df_trans = categories_df_trans['categories'].str.split(";",expand=True)
    row = categories_df_trans.iloc[0,:]
    
    category_colnames = row.apply(lambda x: x.split('-')[0]).values
    categories_df_trans.columns = category_colnames
    
    for column in categories_df_trans:
        # set each value to be the last character of the string
        categories_df_trans[column] = categories_df_trans[column].apply(lambda x: x.split('-')[1])
    
        # convert column from string to numeric
        categories_df_trans[column] = categories_df_trans[column].apply(lambda x: int(x))
    
    #Re-assign ids
    categories_df_trans['id'] = ids
    
    return categories_df_trans

def transform_data(categories_df, messages_df):
    """
    Performs all the necessary data transformations.
    
    Arguments:
        categories_df: categories dataframe
        messages_df: messages dataframe
    
    Output:
        df: the transformed dataframe
    """
    categories_df_trans = transform_categories(categories_df)
    df = pd.merge(messages_df, categories_df_trans, how='inner', on='id')
    df = df[~df.duplicated()]
    return df

def load_data(messages_filepath, categories_filepath):
    """
    Reads the raw data saved as csv files and convert it
    into dataframes.
    
    Arguments:
        messages_filepath: path to the messages.csv file
        categories_filepath: path to the categories.csv file
        
    Output:
        messages_df: messages dataframe
        categories_df: categories dataframe
    """
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)

    return messages_df, categories_df


def save_data(df, database_filename):
    """
    Saves the clean modeling data into a SQLite database.
    
    Arguments:
        df: the transformed dataframe
        database_filename: the resulting database name
        
    Output:
        None
    """
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql(name=database_filename.split('/')[1].split('.')[0],
             con=engine, 
             if_exists = 'replace', 
             index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages_df, categories_df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = transform_data(categories_df, messages_df)
        
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