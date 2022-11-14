import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Get data from csv files and merge them into one dataframe

    Args:
        messages_filepath (_type_): _description_
        categories_filepath (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Load messages
    messages = pd.read_csv(messages_filepath, sep=',')
    # Load categories
    categories = pd.read_csv(categories_filepath, sep=',')

    # Merge datasets
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    """
    Clean data

    Args:
        df (pandas dataframe): dataframe to be cleaned

    Returns:
        pandas dataframe: cleaned dataframe
    """

    # Create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(pat=";", expand=True)

    # Use first row to extract a list of new column names for categories.
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split("-")[0])
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # Drop values that are not 0 or 1
        categories = categories[categories[column].isin(["0", "1"])]
        # Convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Replace categories column in df with new category columns.
    df.drop(columns=["categories"], inplace=True)
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates.
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save data to database

    Args:
        df (pandas dataframe): dataframe to be saved
        database_filename (string): SQL database filename

    """

    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql("DRTable", engine, index=False, if_exists="replace")


def main():
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