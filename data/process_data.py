import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    import messages and categories datasets and merge the together

    INPUT
        messages_filepath   - string containing the messages csv path location
        categories_filepath - string containing the categories csv path location

    OUTPUT
        df - dataframe with the two datased merged

    '''
    messages = pd.read_csv(messages_filepath)

    categories = pd.read_csv(categories_filepath)

    return categories.merge(messages, how = 'left', on = 'id')


def clean_data(df):
    '''
    creates a dataframe of the 36 individual category columns
    renames the new columns
    sets each value to be the last character of the string
    converts column from string to numeric
    concats categoeries df to the main df
    drops dublicates rows

    INPUT
        df - dataframe you want to clean

    OUTPUT
        df - dataframe cleaned

    '''



    categories = df['categories'].str.split(pat = ';', expand = True)

    row = categories.head(1)
    lista = row.values.tolist()
    category_colnames = [x[:-2] for x in lista[0]]
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)

    categories.iloc[:,0].replace(2,1, inplace=True)

    df.drop(['categories'],axis =1, inplace = True)

    df = pd.concat([df,categories], axis=1)



    df.drop_duplicates(inplace = True)

    return df

def save_data(df, database_filename):
    '''
    saves the df in a sqlite database

    INPUT
        df - dataset you want to save
        database_filename - string with the name you want to give to the new file
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('df', engine, index=False,if_exists = 'replace')


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
